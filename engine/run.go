package engine

import (
	"fmt"
	"math"
	"os"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Solver globals
var (
	Time                    float64                      // time in seconds
	alarm                   float64                      // alarm clock marks end time of run, dt adaptation must not cross it!
	pause                   = true                       // set pause at any time to stop running after the current step
	postStep                []func()                     // called on after every full time step
	Inject                           = make(chan func()) // injects code in between time steps. Used by web interface.
	Dt_si                   float64  = 1e-15             // time step = dt_si (seconds) *dt_mul, which should be nice float32
	MinDt, MaxDt            float64                      // minimum and maximum time step
	MaxErr                  float64  = 1e-5              // maximum error/step
	Headroom                float64  = 0.8               // solver headroom, (Gustafsson, 1992, Control of Error and Convergence in ODE Solvers)
	LastErr, PeakErr        float64                      // error of last step, highest error ever
	LastTorque              float64                      // maxTorque of last time step
	NSteps, NUndone, NEvals int                          // number of good steps, undone steps
	FixDt                   float64                      // fixed time step?
	stepper                 Stepper                      // generic step, can be EulerStep, HeunStep, etc
	solvertype              int
)

func init() {
	DeclFunc("Run", Run, "Run the simulation for a time in seconds")
	DeclFunc("Steps", Steps, "Run the simulation for a number of time steps")
	DeclFunc("RunWhile", RunWhile, "Run while condition function is true")
	DeclFunc("SetSolver", SetSolver, "Set solver type.<br>1: Euler<br>2: Heun<br>3: Bogacki-Shampine<br>4: Runge-Kutta (RK4)<br>5: Dormand-Prince<br>6: Fehlberg<br>-1: Backward Euler")
	DeclFunc("ClearPostSteps", func() { postStep = nil }, "Clear the postStep array, which contains functions that are executed after each solver step. This includes running averages, centering routines to track skyrmions and domain walls etc.")
	DeclTVar("t", &Time, "Total simulated time (s)")
	DeclVar("step", &NSteps, "Total number of time steps taken")
	DeclVar("MinDt", &MinDt, "Minimum time step the solver can take (s)")
	DeclVar("MaxDt", &MaxDt, "Maximum time step the solver can take (s)")
	DeclVar("MaxErr", &MaxErr, "Maximum error per step the solver can tolerate (default = 1e-5)")
	DeclVar("Headroom", &Headroom, "Solver headroom (default = 0.8)")
	DeclVar("FixDt", &FixDt, "Set a fixed time step, 0 disables fixed step (which is the default)")
	DeclFunc("Exit", Exit, "Exit from the program")
	SetSolver(DORMANDPRINCE)
	_ = NewScalarValue("dt", "s", "Time Step", func() float64 { return Dt_si })
	_ = NewScalarValue("LastErr", "", "Error of last step", GetLastErr)
	_ = NewScalarValue("PeakErr", "", "Overall maxium error per step", GetPeakErr)
	_ = NewScalarValue("NEval", "", "Total number of torque evaluations", func() float64 { return float64(NEvals) })
}

// Time stepper like Euler, Heun, RK23
type Stepper interface {
	Step() // take time step using solver globals
	Free() // free resources, if any (e.g.: RK23 previous torque)
}

// Arguments for SetSolver
const (
	BACKWARD_EULER  = -1
	EULER           = 1
	HEUN            = 2
	BOGACKISHAMPINE = 3
	RUNGEKUTTA      = 4
	DORMANDPRINCE   = 5
	FEHLBERG        = 6
)

func SetSolver(typ int) {
	// free previous solver, if any
	if stepper != nil {
		stepper.Free()
	}
	switch typ {
	default:
		util.Fatalf("SetSolver: unknown solver type: %v", typ)
	case BACKWARD_EULER:
		stepper = new(BackwardEuler)
	case EULER:
		stepper = new(Euler)
	case HEUN:
		stepper = new(Heun)
	case BOGACKISHAMPINE:
		stepper = new(RK23)
	case RUNGEKUTTA:
		stepper = new(RK4)
	case DORMANDPRINCE:
		stepper = new(RK45DP)
	case FEHLBERG:
		stepper = new(RK56)
	}
	solvertype = typ
}

// write torque to dst and increment NEvals
func torqueFn(dst *data.Slice) {
	SetTorque(dst)
	NEvals++
}

// returns number of torque evaluations
func getNEval() int {
	return NEvals
}

// update lastErr and peakErr
func setLastErr(err float64) {
	LastErr = err
	if err > PeakErr {
		PeakErr = err
	}
}

// Reported reductions stay in persistent GPU slots. The latest torque and
// fixed-step error overwrite their slots in queue order; a separate error slot
// accumulates the exact maximum across steps. This removes the former
// 32-Pending backlog and its mandatory full drain every 16 fixed-dt steps.
var reportedReductions struct {
	torque, fixedErr          *cuda.MaxTracker
	torquePending, errPending bool
	errScale                  float64
}

func torqueReduction() *cuda.MaxTracker {
	if reportedReductions.torque == nil {
		reportedReductions.torque = cuda.NewMaxTracker()
	}
	return reportedReductions.torque
}

func fixedErrReduction() *cuda.MaxTracker {
	if reportedReductions.fixedErr == nil {
		reportedReductions.fixedErr = cuda.NewMaxTracker()
	}
	return reportedReductions.fixedErr
}

// resolveReductions reads the persistent diagnostic slots. The first copy
// drains all earlier GPU work; the slots are then ready to be overwritten by
// the next run segment.
func resolveReductions() {
	if reportedReductions.errPending {
		last, peak := reportedReductions.fixedErr.Values(true)
		setLastErr(last * reportedReductions.errScale)
		peak *= reportedReductions.errScale
		if peak > PeakErr {
			PeakErr = peak
		}
		reportedReductions.errPending = false
	}
	if reportedReductions.torquePending {
		LastTorque, _ = reportedReductions.torque.Values(false)
		reportedReductions.torquePending = false
	}
}

// GetLastErr, GetPeakErr and GetLastTorque are the read side of the deferred
// reductions above: asking for the value is what pays for it.
func GetLastErr() float64 {
	resolveReductions()
	return LastErr
}

func GetPeakErr() float64 {
	resolveReductions()
	return PeakErr
}

func GetLastTorque() float64 {
	resolveReductions()
	return LastTorque
}

// prepareFixedErr starts or continues a fixed-scale peak segment. A scale
// change normally happens between Run calls, after resolveReductions; handle a
// direct Go caller safely as well.
func prepareFixedErr(scale float64) *cuda.MaxTracker {
	if reportedReductions.errPending && reportedReductions.errScale != scale {
		resolveReductions()
	}
	reportedReductions.errScale = scale
	reportedReductions.errPending = true
	return fixedErrReduction()
}

func setLastErrNormLater(v *data.Slice, scale float64) {
	prepareFixedErr(scale).TrackMaxVecNorm(v, true)
}

func setLastErrDiffLater(x, y *data.Slice, scale float64) {
	prepareFixedErr(scale).TrackMaxVecDiff(x, y, true)
}

func setMaxTorque(τ *data.Slice) {
	torqueReduction().TrackMaxVecNorm(τ, false)
	reportedReductions.torquePending = true
}

// adapt time step: dt *= corr, but limited to sensible values.
func adaptDt(corr float64) {
	if FixDt != 0 {
		Dt_si = FixDt
		return
	}

	// corner case triggered by err = 0: just keep time step.
	// see test/regression017.mx3
	if math.IsNaN(corr) {
		corr = 1
	}

	util.AssertMsg(corr != 0, "Time step too small, check if parameters are sensible")
	corr *= Headroom
	if corr > 2 {
		corr = 2
	}
	if corr < 1./2. {
		corr = 1. / 2.
	}
	Dt_si *= corr
	if MinDt != 0 && Dt_si < MinDt {
		Dt_si = MinDt
	}
	if MaxDt != 0 && Dt_si > MaxDt {
		Dt_si = MaxDt
	}
	if Dt_si == 0 {
		util.Fatal("time step too small")
	}

	// do not cross alarm time
	if Time < alarm && Time+Dt_si > alarm {
		Dt_si = alarm - Time
	}

	util.AssertMsg(Dt_si > 0, fmt.Sprint("Time step too small: ", Dt_si))
}

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	stop := Time + seconds
	alarm = stop // don't have dt adapt to go over alarm
	RunWhile(func() bool { return Time < stop })
}

// Run the simulation for a number of steps.
func Steps(n int) {
	stop := NSteps + n
	RunWhile(func() bool { return NSteps < stop })
}

// Runs as long as condition returns true, saves output.
func RunWhile(condition func() bool) {
	SanityCheck()
	pause = false // may be set by <-Inject
	const output = true
	stepper.Free() // start from a clean state
	runWhile(condition, output)
	resolveReductions() // hand the script up-to-date LastErr/PeakErr/LastTorque
	pause = true
}

func runWhile(condition func() bool, output bool) {
	DoOutput() // allow t=0 output
	for condition() && !pause {
		select {
		default:
			step(output)
		// accept tasks form Inject channel
		case f := <-Inject:
			// Injected code, and the gui refresh that rides along with it,
			// reads the reported solver values, so settle them here where we
			// are still on the solver goroutine.
			resolveReductions()
			f()
		}
	}
}

// Runs as long as browser is connected to gui.
func RunInteractive() {
	gui_.RunInteractive()
}

// take one time step
func step(output bool) {
	stepper.Step()
	for _, f := range postStep {
		f()
	}
	if output {
		DoOutput()
	}
}

// Register function f to be called after every time step.
// Typically used, e.g., to manipulate the magnetization.
func PostStep(f func()) {
	postStep = append(postStep, f)
}

// inject code into engine and wait for it to complete.
func InjectAndWait(task func()) {
	ready := make(chan int)
	Inject <- func() { task(); ready <- 1 }
	<-ready
}

func SanityCheck() {
	if Msat.isZero() {
		util.Log("Note: Msat = 0")
	}
	if Aex.isZero() {
		util.Log("Note: Aex = 0")
	}
}

func Exit() {
	Close()
	os.Exit(0)
}
