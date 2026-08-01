package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Implicit midpoint solver.
type BackwardEuler struct {
	dy1 *data.Slice
}

// Euler method, can be used as solver.Step.
func (s *BackwardEuler) Step() {
	util.AssertMsg(MaxErr > 0, "Backward euler solver requires MaxErr > 0")

	t0 := Time

	y := M.Buffer()

	y0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(y0)
	data.Copy(y0, y)

	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	if s.dy1 == nil {
		s.dy1 = cuda.Buffer(VECTOR, y.Size())
		// The predictor below reads this as "the previous torque", but on the
		// first step there is none and Buffer hands out whatever the pool last
		// held. That made the first step, and therefore the whole trajectory,
		// depend on unrelated allocation history: the same script produced
		// different answers depending on which solvers ran before it. Zero
		// means "no predictor", which is what the Temp != 0 path already does.
		cuda.Zero(s.dy1)
	}
	dy1 := s.dy1

	Dt_si = FixDt
	dt := float32(Dt_si * GammaLL)
	util.AssertMsg(dt > 0, "Backward Euler solver requires fixed time step > 0")

	// First guess
	Time = t0 + 0.5*Dt_si // 0.5 dt makes it implicit midpoint method

	// With temperature, previous torque cannot be used as predictor
	if Temp.isZero() {
		cuda.Madd2(y, y0, dy1, 1, dt) // predictor Euler step with previous torque
		M.normalize()
	}

	torqueFn(dy0)
	cuda.Madd2(y, y0, dy0, 1, dt) // y = y0 + dt * dy
	M.normalize()

	// One iteration
	torqueFn(dy1)
	cuda.Madd2(y, y0, dy1, 1, dt) // y = y0 + dt * dy1
	M.normalize()

	Time = t0 + Dt_si

	// This solver always runs at a pinned dt, so the residual between the two
	// iterations is reported but never acted on. Reading it back would drain
	// the GPU pipeline once per step for a number nothing consumes.
	NSteps++
	setLastErrLater(cuda.MaxVecDiffAsync(dy0, dy1), float64(dt))
	setMaxTorque(dy1)
}

func (s *BackwardEuler) Free() {
	s.dy1.Free()
	s.dy1 = nil
}
