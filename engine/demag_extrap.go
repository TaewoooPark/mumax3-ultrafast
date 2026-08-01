package engine

// Experimental polynomial extrapolation of the demagnetizing field.
//
// The implementation follows Eq. (6) of S. Lepadatu,
// IEEE Trans. Magn. 58, 1 (2022), doi:10.1109/TMAG.2022.3159849.
// Exact demagnetizing fields at accepted-step start times are stored with the
// local (zero-displacement) demagnetizing contribution removed. At Runge--Kutta
// substages that non-local field is extrapolated in time with a polynomial of
// the same order as the solver, and the local contribution of the current
// magnetization is added back exactly.

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

var (
	// DemagExtrapolation is deliberately opt-in: it changes the numerical
	// method and introduces a workload-dependent approximation error, even
	// though the matched-order construction preserves the RK method's
	// asymptotic error order.
	DemagExtrapolation = false

	// Read-only diagnostics. "Active" describes the current solver step, not
	// merely the user's requested setting. Unsafe situations are failed closed.
	DemagExtrapolationActive = false
	DemagExtrapolationStatus = "disabled"
	DemagExactEvals          int
	DemagExtrapolatedEvals   int
	DemagRejectedAttempts    int
)

func init() {
	DeclVar("DemagExtrapolation", &DemagExtrapolation,
		"Experimental approximate demagnetizing-field extrapolation for solver 4 (RK4), 5 (Dormand-Prince), or 6 (Fehlberg) only (default=false). This changes numerical results; validate trajectory and energy against DemagExtrapolation=false for each workload")
	DeclFunc("IsDemagExtrapolationActive", func() bool { return DemagExtrapolationActive },
		"Report whether demagnetizing-field extrapolation is active for the current solver step")
	DeclFunc("GetDemagExtrapolationStatus", func() string { return DemagExtrapolationStatus },
		"Report the current demagnetizing-field extrapolation status or safety-disable reason")
	DeclFunc("GetDemagExactEvals", func() int { return DemagExactEvals },
		"Report exact demagnetizing-field convolutions performed while extrapolation was active")
	DeclFunc("GetDemagExtrapolatedEvals", func() int { return DemagExtrapolatedEvals },
		"Report demagnetizing-field convolutions replaced by polynomial extrapolation")
	DeclFunc("GetDemagRejectedAttempts", func() int { return DemagRejectedAttempts },
		"Report rejected solver attempts handled while demagnetizing-field extrapolation was active")
	DeclFunc("ResetDemagExtrapolation", ResetDemagExtrapolation,
		"Discard all demagnetizing-field extrapolation history and counters")
}

type demagHistorySample struct {
	time  float64
	field *data.Slice // exact field with the local/self contribution removed
}

type demagExtrapConfig struct {
	size           [3]int
	cellSize       [3]float64
	pbc            [3]int
	accuracy       float64
	msatRevision   uint64
	maskRevision   uint64
	geomRevision   uint64
	regionRevision uint64
}

type demagExtrapolator struct {
	history []demagHistorySample // oldest to newest, accepted-step states only
	pending *demagHistorySample  // exact current step-start state; committed on accept

	order        int
	stepTime     float64
	active       bool
	sawStepExact bool

	config      demagExtrapConfig
	configValid bool
	lastLog     string
}

var demagExtrap demagExtrapolator

// demagSelfCoeff is the zero-displacement diagonal of the actual convolution
// kernel. For PBC this deliberately includes the periodic images folded into
// that coefficient: it is still a purely local, exactly computable term.
var (
	demagSelfCoeff      [3]float32
	demagSelfCoeffReady bool
)

func setDemagSelfCoeff(kernel [3][3]*data.Slice) {
	for c := 0; c < 3; c++ {
		demagSelfCoeff[c] = kernel[c][c].Scalars()[0][0][0]
	}
	demagSelfCoeffReady = true
}

func clearDemagSelfCoeff() {
	demagSelfCoeff = [3]float32{}
	demagSelfCoeffReady = false
}

// ResetDemagExtrapolation is public so Go integrations that mutate M.Buffer()
// directly have a safe, explicit invalidation hook.
func ResetDemagExtrapolation() {
	demagExtrap.resetHistory()
	DemagExactEvals = 0
	DemagExtrapolatedEvals = 0
	DemagRejectedAttempts = 0
	DemagExtrapolationActive = false
	if DemagExtrapolation {
		DemagExtrapolationStatus = "history reset"
	} else {
		DemagExtrapolationStatus = "disabled"
	}
}

func invalidateDemagExtrapolation() {
	demagExtrap.resetHistory()
	DemagExtrapolationActive = false
	if DemagExtrapolation {
		DemagExtrapolationStatus = "history invalidated"
	}
}

func (d *demagExtrapolator) resetHistory() {
	for i := range d.history {
		if d.history[i].field != nil {
			d.history[i].field.Free()
		}
	}
	d.history = nil
	if d.pending != nil && d.pending.field != nil {
		d.pending.field.Free()
	}
	d.pending = nil
	d.order = 0
	d.stepTime = 0
	d.active = false
	d.sawStepExact = false
	d.configValid = false
}

func demagExtrapolationOrder() int {
	switch solvertype {
	case HEUN:
		return 2
	case BOGACKISHAMPINE:
		return 3
	case RUNGEKUTTA:
		return 4
	case DORMANDPRINCE, FEHLBERG:
		// The accepted members of both embedded pairs are fifth order for
		// purposes of the field approximation used in the reference method.
		return 5
	default:
		return 0
	}
}

func demagExtrapolationSolverSupported() bool {
	return demagExtrapolationOrder() >= 4
}

func (d *demagExtrapolator) currentConfig() demagExtrapConfig {
	m := Mesh()
	return demagExtrapConfig{
		size:           m.Size(),
		cellSize:       m.CellSize(),
		pbc:            m.PBC(),
		accuracy:       DemagAccuracy,
		msatRevision:   Msat.revision,
		maskRevision:   NoDemagSpins.revision,
		geomRevision:   geometry.revision,
		regionRevision: regions.revision,
	}
}

func (d *demagExtrapolator) disable(reason string) {
	d.resetHistory()
	DemagExtrapolationActive = false
	DemagExtrapolationStatus = reason
	d.logOnce(reason)
}

func (d *demagExtrapolator) logOnce(message string) {
	if message == d.lastLog {
		return
	}
	d.lastLog = message
	util.Log("DemagExtrapolation: " + message)
}

func (d *demagExtrapolator) beginStep() {
	d.active = false
	d.sawStepExact = false
	DemagExtrapolationActive = false

	if !DemagExtrapolation {
		if len(d.history) != 0 || d.pending != nil {
			d.resetHistory()
		}
		DemagExtrapolationStatus = "disabled"
		return
	}
	if !EnableDemag {
		d.disable("disabled: EnableDemag=false")
		return
	}
	if relaxing {
		d.disable("disabled: relaxation/minimization")
		return
	}
	if len(postStep) != 0 {
		d.disable("disabled: postStep callbacks may mutate state")
		return
	}
	if Temp.hasTimeDependence() || !Temp.isZero() {
		d.disable("disabled: finite or time-dependent temperature")
		return
	}
	if Msat.hasTimeDependence() {
		d.disable("disabled: time-dependent Msat")
		return
	}
	if NoDemagSpins.hasTimeDependence() || !NoDemagSpins.isZero() {
		d.disable("disabled: NoDemagSpins mask")
		return
	}

	order := demagExtrapolationOrder()
	if order == 0 {
		d.disable("disabled: solver is not a supported explicit RK method")
		return
	}
	// Low-order extrapolation was unstable under coarse-step stress tests.
	// This is a hard safety boundary, not a user-tunable policy: in particular,
	// Heun and Bogacki--Shampine cannot opt in through another setting.
	if !demagExtrapolationSolverSupported() {
		d.disable("disabled: demag extrapolation requires solver 4 (RK4), 5 (Dormand-Prince), or 6 (Fehlberg); lower-order solvers are hard-disabled")
		return
	}

	cfg := d.currentConfig()
	if !d.configValid || d.config != cfg || d.order != order {
		d.resetHistory()
		d.config = cfg
		d.configValid = true
		d.order = order
	}

	// A pending sample is retained only across a rejected attempt at exactly
	// the same accepted state. Any other time discontinuity is an explicit
	// state change from the extrapolator's point of view.
	if d.pending != nil && d.pending.time != Time {
		d.resetHistory()
		d.config = cfg
		d.configValid = true
		d.order = order
	}
	if len(d.history) != 0 {
		last := d.history[len(d.history)-1].time
		if !(Time > last) {
			d.resetHistory()
			d.config = cfg
			d.configValid = true
			d.order = order
		}
	}

	d.stepTime = Time
	d.active = true
	DemagExtrapolationActive = true
	if len(d.history)+btoi(d.pending != nil) < order+1 {
		DemagExtrapolationStatus = "priming exact accepted-step history; later demag substages are approximate"
	} else {
		DemagExtrapolationStatus = "active: approximate demag substages; workload-specific exact A/B validation required"
	}
	d.logOnce("enabled experimental approximation for solver order >=4; validate trajectory and energy against DemagExtrapolation=false")
	Refer("lepadatu2022")
}

func btoi(v bool) int {
	if v {
		return 1
	}
	return 0
}

func demagExtrapolationStepActive() bool {
	return demagExtrap.active
}

// tryDemagExtrapolation returns true when dst has been filled without a full
// convolution. The first demag field of every step remains exact. On a retry,
// the exact field at the restored step-start state can be reconstructed from
// the retained pending non-local field plus the exact current local term.
func tryDemagExtrapolation(dst *data.Slice) bool {
	d := &demagExtrap
	if !d.active {
		return false
	}

	if !d.sawStepExact {
		if d.pending == nil || d.pending.time != d.stepTime {
			return false
		}
		data.Copy(dst, d.pending.field)
		addDemagSelf(dst, 1)
		d.sawStepExact = true
		return true
	}

	need := d.order + 1
	if d.pending == nil || len(d.history)+1 < need {
		return false // exact priming evaluation
	}

	start := len(d.history) - (need - 1)
	if start < 0 {
		return false
	}
	var sampleStorage [6]demagHistorySample
	samples := sampleStorage[:need]
	copy(samples, d.history[start:])
	samples[need-1] = *d.pending
	var weightStorage [6]float32
	weights := weightStorage[:need]
	if !lagrangeWeightsInto(samples, Time, weights) {
		d.disable("disabled: ill-conditioned extrapolation history")
		return false
	}
	interpolateDemag(dst, samples, weights)
	addDemagSelf(dst, 1)
	DemagExtrapolatedEvals++
	return true
}

// observeExactDemag is called after the normal convolution path. Only its
// first result in a step is a trajectory point and is eligible for history.
func observeExactDemag(dst *data.Slice) {
	d := &demagExtrap
	if !d.active {
		return
	}
	DemagExactEvals++
	if d.sawStepExact {
		return
	}
	if Time != d.stepTime || !demagSelfCoeffReady {
		d.disable("disabled: exact step-start demag field unavailable")
		return
	}

	var slot *data.Slice
	maxSamples := d.order + 1
	if len(d.history) >= maxSamples {
		slot = d.history[0].field
		copy(d.history, d.history[1:])
		d.history = d.history[:len(d.history)-1]
	} else {
		slot = cuda.NewSlice(VECTOR, dst.Size())
	}
	data.Copy(slot, dst)
	addDemagSelf(slot, -1)
	d.pending = &demagHistorySample{time: d.stepTime, field: slot}
	d.sawStepExact = true
}

func (d *demagExtrapolator) endStep(accepted bool) {
	if d.active && accepted && d.pending != nil {
		d.history = append(d.history, *d.pending)
		d.pending = nil
	} else if d.active && !accepted {
		DemagRejectedAttempts++
	}
	d.active = false
	d.sawStepExact = false
	DemagExtrapolationActive = false
}

func lagrangeWeights(samples []demagHistorySample, t float64) ([]float32, bool) {
	n := len(samples)
	w := make([]float32, n)
	return w, lagrangeWeightsInto(samples, t, w)
}

func lagrangeWeightsInto(samples []demagHistorySample, t float64, w []float32) bool {
	n := len(samples)
	if n == 0 {
		return false
	}
	if n > 6 || len(w) != n {
		return false
	}
	// Normalize time before forming products. Simulation times can be ~1e-9
	// while accepted-step separations are ~1e-15; subtracting once in float64
	// and working with O(1) abscissae avoids needless product underflow and
	// improves conditioning for adaptive steps.
	origin := samples[n-1].time
	span := origin - samples[0].time
	if !(span > 0) || math.IsNaN(span) || math.IsInf(span, 0) {
		return false
	}
	x := (t - origin) / span
	var xi [6]float64
	for i := range samples {
		xi[i] = (samples[i].time - origin) / span
		if i != 0 && !(xi[i] > xi[i-1]) {
			return false
		}
	}

	for i := 0; i < n; i++ {
		wi := 1.0
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			den := xi[i] - xi[j]
			if den == 0 {
				return false
			}
			wi *= (x - xi[j]) / den
		}
		if math.IsNaN(wi) || math.IsInf(wi, 0) || math.Abs(wi) > 1e6 {
			return false
		}
		w[i] = float32(wi)
	}
	return true
}

func interpolateDemag(dst *data.Slice, samples []demagHistorySample, w []float32) {
	switch len(samples) {
	case 3:
		cuda.Madd3(dst,
			samples[0].field, samples[1].field, samples[2].field,
			w[0], w[1], w[2])
	case 4:
		cuda.Madd4(dst,
			samples[0].field, samples[1].field, samples[2].field, samples[3].field,
			w[0], w[1], w[2], w[3])
	case 5:
		cuda.Madd5(dst,
			samples[0].field, samples[1].field, samples[2].field, samples[3].field, samples[4].field,
			w[0], w[1], w[2], w[3], w[4])
	case 6:
		cuda.Madd6(dst,
			samples[0].field, samples[1].field, samples[2].field, samples[3].field, samples[4].field, samples[5].field,
			w[0], w[1], w[2], w[3], w[4], w[5])
	default:
		panic("unsupported demagnetizing-field extrapolation order")
	}
}

// addDemagSelf adds sign*K(0)*mu0*Msat*geometry*M to field. The coefficient
// and every spatial factor are exactly those used by the convolution input.
// It intentionally uses existing cross-backend kernels; the extra elementwise
// work is much cheaper than another padded FFT and keeps CUDA compatibility.
func addDemagSelf(field *data.Slice, sign float32) {
	if !demagSelfCoeffReady {
		return
	}
	m := M.Buffer()
	geom := geometry.Gpu()
	mu0 := float32(mag.Mu0)

	if Msat.IsUniform() {
		ms := float32(Msat.GetRegion(0))
		if geom.IsNil() {
			for c := 0; c < 3; c++ {
				fac := sign * demagSelfCoeff[c] * (mu0 * ms)
				cuda.Madd2(field.Comp(c), field.Comp(c), m.Comp(c), 1, fac)
			}
			return
		}

		tmp := cuda.Buffer(SCALAR, m.Size())
		defer cuda.Recycle(tmp)
		for c := 0; c < 3; c++ {
			cuda.Mul(tmp, m.Comp(c), geom)
			fac := sign * demagSelfCoeff[c] * (mu0 * ms)
			cuda.Madd2(field.Comp(c), field.Comp(c), tmp, 1, fac)
		}
		return
	}

	ms, recycle := Msat.Slice()
	if recycle {
		defer cuda.Recycle(ms)
	}
	if !geom.IsNil() {
		cuda.Mul(ms, ms, geom)
	}
	tmp := cuda.Buffer(SCALAR, m.Size())
	defer cuda.Recycle(tmp)
	for c := 0; c < 3; c++ {
		cuda.Mul(tmp, m.Comp(c), ms)
		fac := sign * demagSelfCoeff[c] * mu0
		cuda.Madd2(field.Comp(c), field.Comp(c), tmp, 1, fac)
	}
}
