package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

type RK45DP struct {
	k1 *data.Slice // torque at end of step is kept for beginning of next step

	// Speculative pipelining state, used only while SpeculativeStep is on.
	//
	// A step is encoded and its error estimate is left in flight, so the host
	// can encode the following step while the GPU is still executing this one.
	// Everything needed to undo the step lives here until the estimate is read.
	// k1Alt and the two m0 slots rotate rather than being copied: the backup and
	// the FSAL handover already happen once per step, they are just directed at
	// a slot the unsettled step does not own.
	pending   *cuda.Pending
	pendingH  float32     // internal step size that scales the estimate
	pendingT0 float64     // Time before the unsettled step
	pendingDt float64     // Dt_si the unsettled step was taken with
	pendingM0 *data.Slice // m before the unsettled step
	pendingK1 *data.Slice // k1 before the unsettled step
	k1Alt     *data.Slice // FSAL slot not owned by the unsettled step
	m0        [2]*data.Slice
	m0Next    int
}

func (rk *RK45DP) Step() {
	m := M.Buffer()
	size := m.Size()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
	}

	speculate := speculativeStepEligible()
	if !speculate {
		// The conditions a pending step was encoded under no longer hold, so it
		// has to be judged before anything else happens.
		rk.Settle()
	}

	// first step ever: one-time k1 init and eval
	initializedK1 := false
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		torqueFn(rk.k1)
		initializedK1 = true
	}

	// An exact demag evaluation at the accepted step-start state anchors the
	// extrapolation polynomial. The previous k7 used an extrapolated substage
	// field, so recompute k1 while this experimental path is active. Default
	// FSAL behavior is untouched when extrapolation is disabled.
	if demagExtrapolationStepActive() {
		if !initializedK1 {
			torqueFn(rk.k1)
		}
	} else if !Temp.isZero() { // FSAL cannot be used with finite temperature
		torqueFn(rk.k1)
	}

	t0 := Time
	dt0 := Dt_si
	// backup magnetization
	var m0 *data.Slice
	if speculate {
		// A rollback target has to outlive this call, so it cannot come from the
		// recycled pool.
		m0 = rk.stepStartSlot(size)
	} else {
		m0 = cuda.Buffer(3, size)
		defer cuda.Recycle(m0)
	}
	data.Copy(m0, m)

	k2, k3, k4, k5, k6 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	// k2 will be re-used as k7

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL

	// there is no explicit stage 1: k1 from previous step

	// stage 2
	Time = t0 + (1./5.)*Dt_si
	cuda.Madd2(m, m, rk.k1, 1, (1./5.)*h) // m = m*1 + k1*h/5
	M.normalize()
	torqueFn(k2)

	// stage 3
	Time = t0 + (3./10.)*Dt_si
	cuda.Madd3(m, m0, rk.k1, k2, 1, (3./40.)*h, (9./40.)*h)
	M.normalize()
	torqueFn(k3)

	// stage 4
	Time = t0 + (4./5.)*Dt_si
	cuda.Madd4(m, m0, rk.k1, k2, k3, 1, (44./45.)*h, (-56./15.)*h, (32./9.)*h)
	M.normalize()
	torqueFn(k4)

	// stage 5
	Time = t0 + (8./9.)*Dt_si
	cuda.Madd5(m, m0, rk.k1, k2, k3, k4, 1, (19372./6561.)*h, (-25360./2187.)*h, (64448./6561.)*h, (-212./729.)*h)
	M.normalize()
	torqueFn(k5)

	// stage 6
	Time = t0 + (1.)*Dt_si
	cuda.Madd6(m, m0, rk.k1, k2, k3, k4, k5, 1, (9017./3168.)*h, (-355./33.)*h, (46732./5247.)*h, (49./176.)*h, (-5103./18656.)*h)
	M.normalize()
	torqueFn(k6)

	// stage 7: 5th order solution
	Time = t0 + (1.)*Dt_si
	// no k2
	cuda.Madd6(m, m0, rk.k1, k3, k4, k5, k6, 1, (35./384.)*h, (500./1113.)*h, (125./192.)*h, (-2187./6784.)*h, (11./84.)*h) // 5th
	M.normalize()
	k7 := k2     // re-use k2
	torqueFn(k7) // next torque if OK

	// error estimate
	Err := k3 // k3 is dead after the final solution; reuse it for the error
	cuda.Madd6(Err, rk.k1, k3, k4, k5, k6, k7, (35./384.)-(5179./57600.), (500./1113.)-(7571./16695.), (125./192.)-(393./640.), (-2187./6784.)-(-92097./339200.), (11./84.)-(187./2100.), (0.)-(1./40.))

	// determine error
	//
	// A pinned dt makes the step unconditional, so the estimate is reported
	// but never waited for: reading it back would drain the GPU pipeline for a
	// number nothing acts on.
	if FixDt != 0 {
		setLastErrNormLater(Err, float64(h))
		setMaxTorque(k7)
		NSteps++
		Time = t0 + Dt_si
		Dt_si = FixDt // what adaptDt does under a pinned step
		data.Copy(rk.k1, k7)
		return
	}

	if speculate {
		rk.stepSpeculative(size, m, m0, k7, Err, h, t0, dt0)
		return
	}

	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt { // mindt check to avoid infinite loop
		// step OK
		setLastErr(err)
		setMaxTorque(k7)
		NSteps++
		Time = t0 + Dt_si
		adaptDt(math.Pow(MaxErr/err, 1./5.))
		data.Copy(rk.k1, k7) // FSAL
	} else {
		// undo bad step
		//util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	}
}

// stepSpeculative finishes a step without waiting for its error estimate.
//
// The estimate is enqueued, then the step encoded one call earlier is judged.
// That step's work sits ahead of this one in the queue, so the wait for it
// overlaps GPU work instead of standing in an empty pipeline - which is the
// whole point: at 128x128 the host spends longer encoding a step than the GPU
// spends running it, and a drain per step forces the two to alternate.
//
// Only if the earlier step survives does this one get provisionally accepted.
func (rk *RK45DP) stepSpeculative(size [3]int, m, m0, k7, Err *data.Slice, h float32, t0, dt0 float64) {
	// Targeted: close the queue batch here so the next step lands in a fresh one.
	// Without that boundary, waiting for this estimate would also wait for the
	// step encoded after it and the pipelining would buy nothing.
	pending := cuda.MaxVecNormAsync(Err).Targeted()

	if !rk.settlePending() {
		// The earlier step was rejected and the magnetization has been rewound
		// past it, so this step was built on a state that no longer exists.
		// Reading the estimate releases its reduction slots; the value is moot.
		pending.Value()
		return
	}

	// Provisionally accept. FSAL goes to the slot the judged step has released,
	// which keeps the previous torque intact as a rollback target.
	startK1 := rk.k1
	next := rk.fsalSlot(size)
	data.Copy(next, k7)
	rk.k1, rk.k1Alt = next, startK1

	setMaxTorque(k7) // reported before the step is judged, so a rejected step may show
	NSteps++
	Time = t0 + dt0

	rk.pending = pending
	rk.pendingH = h
	rk.pendingT0 = t0
	rk.pendingDt = dt0
	rk.pendingM0 = m0
	rk.pendingK1 = startK1
}

// Settle judges a step left in flight, so that m, Time and NSteps stop being
// provisional. It is idempotent and cheap when nothing is pending.
func (rk *RK45DP) Settle() {
	rk.settlePending()
}

// settlePending reads the pending error estimate and either confirms the
// provisional acceptance or rewinds it. It reports whether the step was kept.
func (rk *RK45DP) settlePending() bool {
	if rk.pending == nil {
		return true
	}
	err := rk.pending.Value() * float64(rk.pendingH)
	dt := rk.pendingDt
	m0 := rk.pendingM0
	k1 := rk.pendingK1
	t0 := rk.pendingT0
	rk.pending = nil
	rk.pendingM0 = nil
	rk.pendingK1 = nil

	// The controller scales the step that produced this estimate, which is one
	// step behind the one just encoded. Restoring it first makes the decision
	// identical to the exact controller's, only taken a step later.
	Dt_si = dt

	if err < MaxErr || dt <= MinDt { // mindt check to avoid infinite loop
		setLastErr(err)
		adaptDt(math.Pow(MaxErr/err, 1./5.))
		return true
	}

	// Undo the provisional acceptance.
	util.Assert(FixDt == 0)
	Time = t0
	data.Copy(M.Buffer(), m0)
	if k1 != nil {
		rk.k1Alt = rk.k1
		rk.k1 = k1
	}
	NSteps--
	NUndone++
	adaptDt(math.Pow(MaxErr/err, 1./6.))
	return false
}

// stepStartSlot hands back the backup slot the unsettled step does not own, so
// its rollback target survives while the next step is encoded.
func (rk *RK45DP) stepStartSlot(size [3]int) *data.Slice {
	if rk.m0[0] == nil {
		rk.m0[0] = cuda.NewSlice(3, size)
		rk.m0[1] = cuda.NewSlice(3, size)
	}
	slot := rk.m0[rk.m0Next]
	if slot == rk.pendingM0 {
		rk.m0Next = 1 - rk.m0Next
		slot = rk.m0[rk.m0Next]
	}
	rk.m0Next = 1 - rk.m0Next
	return slot
}

// fsalSlot hands back the spare FSAL slot, allocating it on first use.
func (rk *RK45DP) fsalSlot(size [3]int) *data.Slice {
	if rk.k1Alt == nil {
		rk.k1Alt = cuda.NewSlice(3, size)
	}
	return rk.k1Alt
}

func (rk *RK45DP) Free() {
	// A pending step is dropped rather than judged: Free is how the solver is
	// reset, including on a resize where the saved state has the wrong size.
	// Callers that need the state intact call Settle first, which runWhile does
	// on the way out of every run loop.
	if rk.pending != nil {
		rk.pending.Value() // releases the reduction slots
		rk.pending = nil
	}
	rk.pendingM0 = nil
	rk.pendingK1 = nil
	rk.k1.Free()
	rk.k1 = nil
	rk.k1Alt.Free()
	rk.k1Alt = nil
	for i := range rk.m0 {
		rk.m0[i].Free()
		rk.m0[i] = nil
	}
	rk.m0Next = 0
}
