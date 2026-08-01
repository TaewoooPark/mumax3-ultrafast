package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).

import (
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	DmSamples             int     = 10   // number of dm to keep for convergence check
	StopMaxDm             float64 = 1e-6 // stop minimizer if sampled dm is smaller than this
	MinimizeWallClockTime float64 = -1.0 // wall-clock time limit for minimization
	MinimizeConverged     bool           // true if minimize converged, and false if the maximum wall-clock time is reached

	// MinimizeOnGPU keeps the Barzilai-Borwein step size in device memory
	// instead of routing it through the host, which is what otherwise costs one
	// full pipeline drain per iteration. Each descent is bit-identical: the
	// kernel re-adds the same partial slots in the same order the host does. The
	// convergence norm is resolved one iteration late, so a minimization can
	// stop one iteration later than it otherwise would - always on the more
	// converged side. Opt-in for that reason.
	MinimizeOnGPU = false
)

func init() {
	DeclFunc("Minimize", Minimize, "Use steepest conjugate gradient method to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded. The wall-clock time limit is disabled by default.")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check.")
	DeclVar("MinimizeWallClockTime", &MinimizeWallClockTime, "Wall-clock time limit (seconds) for Minimize that will interrupt the minimization if exceeded. Set to -1 (default) to disable. An interrupted minimization does not guarantee a correct solution.")
	DeclVar("MinimizeOnGPU", &MinimizeOnGPU, "Keep the Minimize step size in device memory so an iteration does not drain the GPU pipeline (default=false). Each descent is bit-identical; the convergence check is one iteration late, so a minimization may take one extra step.")
}

// fixed length FIFO. Items can be added but not removed
type fifoRing struct {
	count int
	tail  int // index to put next item. Will loop to 0 after exceeding length
	data  []float64
}

func FifoRing(length int) fifoRing {
	return fifoRing{data: make([]float64, length)}
}

func (r *fifoRing) Add(item float64) {
	r.data[r.tail] = item
	r.count++
	r.tail = (r.tail + 1) % len(r.data)
	if r.count > len(r.data) {
		r.count = len(r.data)
	}
}

func (r *fifoRing) Max() float64 {
	max := r.data[0]
	for i := 1; i < r.count; i++ {
		if r.data[i] > max {
			max = r.data[i]
		}
	}
	return max
}

type Minimizer struct {
	k      *data.Slice // torque saved to calculate time step
	lastDm fifoRing
	h      float32

	// Device-resident step size, used when MinimizeOnGPU is on. The step size
	// then never reaches the host, so an iteration no longer has to drain the
	// pipeline before the next descent can be encoded.
	bb        *cuda.BBStep
	dmPending *cuda.Pending // convergence norm of the previous iteration
}

func (mini *Minimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	if mini.k == nil {
		mini.k = cuda.Buffer(3, size)
		torqueFn(mini.k)
	}

	onGPU := minimizeOnGPUEligible()
	if !onGPU {
		mini.Settle()
	}
	if onGPU && mini.bb == nil {
		mini.bb = cuda.NewBBStep()
		mini.bb.Seed(mini.h)
	}

	k := mini.k
	h := mini.h

	// save original magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	// make descent
	if onGPU {
		cuda.MinimizeBB(m, m0, k, mini.bb)
	} else {
		cuda.Minimize(m, m0, k, h)
	}

	// calculate new torque for next step
	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)
	data.Copy(k0, k)
	torqueFn(k)
	setMaxTorque(k) // report to user

	// just to make the following readable
	dm := m0
	dk := k0

	// calculate step difference of m and k
	cuda.Madd2(dm, m, m0, 1., -1.)
	cuda.Madd2(dk, k, k0, -1., 1.) // reversed due to LLNoPrecess sign

	if onGPU {
		// The step size for the next descent stays on the device, so nothing
		// here has to be read back. The convergence norm still has to reach the
		// host eventually, but it only decides when to stop, so it is resolved
		// one iteration late - by which time a whole iteration of GPU work sits
		// behind it and the wait overlaps instead of stalling.
		mini.bb.Update(dm, dk, NSteps%2 == 0)
		pending := cuda.MaxVecNormAsync(dm).Targeted()
		mini.resolveDm()
		mini.dmPending = pending
		M.normalize()
		NSteps++ // as a convention, time does not advance during relax
		return
	}

	// The convergence norm and both BB step-size terms are independent. Queue
	// all three reductions before reading any result so one GPU drain serves the
	// complete minimizer step.
	maxDmPending := cuda.MaxVecNormAsync(dm)
	var nomPending, divPending *cuda.Pending
	if NSteps%2 == 0 {
		nomPending = cuda.DotAsync(dm, dm)
		divPending = cuda.DotAsync(dm, dk)
	} else {
		nomPending = cuda.DotAsync(dm, dk)
		divPending = cuda.DotAsync(dk, dk)
	}
	maxDm := maxDmPending.Value()
	mini.lastDm.Add(maxDm)
	setLastErr(mini.lastDm.Max()) // report maxDm to user as LastErr
	nom := float32(nomPending.Value())
	div := float32(divPending.Value())
	if div != 0. {
		mini.h = nom / div
	} else { // in case of division by zero
		mini.h = 1e-4
	}

	M.normalize()

	// as a convention, time does not advance during relax
	NSteps++
}

// Settle records the convergence norm of an iteration that was left in flight,
// so that the stopping test sees every sample. runWhile calls this on the way out
// of the loop and then re-tests its condition, which is what keeps the criterion
// from being evaluated against a short history.
func (mini *Minimizer) Settle() {
	mini.resolveDm()
}

func (mini *Minimizer) resolveDm() {
	if mini.dmPending == nil {
		return
	}
	maxDm := mini.dmPending.Value()
	mini.dmPending = nil
	mini.lastDm.Add(maxDm)
	setLastErr(mini.lastDm.Max()) // report maxDm to user as LastErr
}

// minimizeOnGPUEligible reports whether the descent may take its step size from
// device memory.
func minimizeOnGPUEligible() bool {
	return MinimizeOnGPU && cuda.BBStepSupported
}

func (mini *Minimizer) Free() {
	if mini.dmPending != nil {
		mini.dmPending.Value() // releases the reduction slots
		mini.dmPending = nil
	}
	mini.bb.Free()
	mini.bb = nil
	mini.k.Free()
}

// helper function that returns false if the wall clock time limit is exceeded. If the wall-clock time is negative, this function always returns true.
func WallclockTimer(start time.Time, WallClockTime float64) bool {
	if WallClockTime < 0 {
		return true
	}
	if WallClockTime == 0 {
		return false
	}
	return time.Since(start) < time.Duration(WallClockTime*float64(time.Second))
}

func Minimize() bool {

	// if wall-clock time is zero, skip minimization entirely (zero steps), and don't change any settings
	MinimizeConverged = false
	TimerStart := time.Now()
	if MinimizeWallClockTime == 0 {
		MinimizeConverged = false
		return MinimizeConverged
	}

	Refer("exl2014")
	SanityCheck()
	// Save the settings we are changing...
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time

	relaxing = true // disable temperature noise

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0

		relaxing = false
	}()

	Precess = false // disable precession for torque calculation
	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the minimizer
	mini := Minimizer{
		h:      1e-4,
		k:      nil,
		lastDm: FifoRing(DmSamples)}
	stepper = &mini

	cond := func() bool {
		return (mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm) && WallclockTimer(TimerStart, MinimizeWallClockTime)
	}

	RunWhile(cond)
	pause = true
	// if the loop ended because of convergence, then MinimizeConverged is true. If the loop ended because of wall-clock time, then MinimizeConverged is false.
	MinimizeConverged = !(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	stepper.Free()
	return MinimizeConverged
}
