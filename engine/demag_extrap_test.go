package engine

import (
	"math"
	"testing"
)

func TestLagrangeWeightsReproduceMatchedOrderPolynomial(t *testing.T) {
	times := []float64{0.0, 0.13, 0.31, 0.58, 0.82, 1.0}
	target := 1.17
	for n := 3; n <= len(times); n++ {
		samples := make([]demagHistorySample, n)
		for i := range samples {
			samples[i].time = times[len(times)-n+i]
		}
		weights, ok := lagrangeWeights(samples, target)
		if !ok {
			t.Fatalf("n=%d: weights rejected", n)
		}
		for degree := 0; degree < n; degree++ {
			got := 0.0
			for i := range samples {
				got += float64(weights[i]) * math.Pow(samples[i].time, float64(degree))
			}
			want := math.Pow(target, float64(degree))
			if math.Abs(got-want) > 2e-5*math.Max(1, math.Abs(want)) {
				t.Fatalf("n=%d degree=%d: got %.12g, want %.12g", n, degree, got, want)
			}
		}
	}
}

func TestLagrangeWeightsRejectRepeatedTimes(t *testing.T) {
	samples := []demagHistorySample{{time: 0}, {time: 1}, {time: 1}}
	if _, ok := lagrangeWeights(samples, 1.5); ok {
		t.Fatal("repeated history time was accepted")
	}
}

func TestRejectedAttemptDoesNotCommitPendingHistory(t *testing.T) {
	oldRejected := DemagRejectedAttempts
	defer func() { DemagRejectedAttempts = oldRejected }()
	d := demagExtrapolator{
		active:  true,
		pending: &demagHistorySample{time: 1},
	}
	d.endStep(false)
	if len(d.history) != 0 {
		t.Fatalf("rejected attempt committed %d history entries", len(d.history))
	}
	if d.pending == nil {
		t.Fatal("exact accepted step-start sample was not retained for retry")
	}
	if DemagRejectedAttempts != oldRejected+1 {
		t.Fatalf("rejected-attempt counter = %d, want %d", DemagRejectedAttempts, oldRejected+1)
	}

	d.active = true
	d.endStep(true)
	if len(d.history) != 1 || d.history[0].time != 1 {
		t.Fatalf("accepted attempt history = %#v", d.history)
	}
	if d.pending != nil {
		t.Fatal("accepted pending sample was not cleared")
	}
	// field is nil in this lifecycle-only test, so reset is allocation-free.
	d.resetHistory()
}

func TestContinuationTimeGuardPreservesAcceptedAndRejectedLifecycles(t *testing.T) {
	oldTime := Time
	oldRejected := DemagRejectedAttempts
	defer func() {
		Time = oldTime
		DemagRejectedAttempts = oldRejected
	}()

	t.Run("accepted continuation", func(t *testing.T) {
		d := demagExtrapolator{
			active:  true,
			pending: &demagHistorySample{time: 1},
		}
		Time = 2
		d.endStep(true)

		if !d.continuationValid || d.continuationTime != 2 {
			t.Fatalf("accepted continuation = (%v, %g), want (true, 2)", d.continuationValid, d.continuationTime)
		}
		if d.resetOnTimeDiscontinuity(2) {
			t.Fatal("normal accepted-step continuation reset history")
		}
		if len(d.history) != 1 || d.history[0].time != 1 {
			t.Fatalf("accepted history changed at continuation: %#v", d.history)
		}

		if !d.resetOnTimeDiscontinuity(12) {
			t.Fatal("external forward time jump did not reset history")
		}
		if len(d.history) != 0 || d.pending != nil || d.continuationValid {
			t.Fatalf("time-jump reset left stale state: %#v", d)
		}
	})

	t.Run("rejected retry", func(t *testing.T) {
		d := demagExtrapolator{
			active:  true,
			history: []demagHistorySample{{time: 0}},
			pending: &demagHistorySample{time: 1},
		}
		Time = 1
		d.endStep(false)

		if !d.continuationValid || d.continuationTime != 1 {
			t.Fatalf("rejected continuation = (%v, %g), want (true, 1)", d.continuationValid, d.continuationTime)
		}
		if d.resetOnTimeDiscontinuity(1) {
			t.Fatal("normal rejected-attempt retry reset history")
		}
		if len(d.history) != 1 || d.pending == nil || d.pending.time != 1 {
			t.Fatalf("rejected retry lost retained state: %#v", d)
		}
		d.resetHistory()
	})
}

func TestDemagExtrapolationOrderMatchesSupportedSolvers(t *testing.T) {
	old := solvertype
	defer func() { solvertype = old }()
	tests := map[int]int{
		EULER:           0,
		HEUN:            2,
		BOGACKISHAMPINE: 3,
		RUNGEKUTTA:      4,
		DORMANDPRINCE:   5,
		FEHLBERG:        5,
		BACKWARD_EULER:  0,
	}
	for solver, want := range tests {
		solvertype = solver
		if got := demagExtrapolationOrder(); got != want {
			t.Errorf("solver %d: order %d, want %d", solver, got, want)
		}
	}
}

func TestDemagExtrapolationHardDisablesLowOrderSolvers(t *testing.T) {
	old := solvertype
	defer func() { solvertype = old }()
	for _, solver := range []int{EULER, HEUN, BOGACKISHAMPINE, BACKWARD_EULER} {
		solvertype = solver
		if demagExtrapolationSolverSupported() {
			t.Errorf("solver %d bypassed the hard order >=4 guard", solver)
		}
	}
	for _, solver := range []int{RUNGEKUTTA, DORMANDPRINCE, FEHLBERG} {
		solvertype = solver
		if !demagExtrapolationSolverSupported() {
			t.Errorf("solver %d was rejected by the order >=4 guard", solver)
		}
	}
}

func TestInactiveExactQueryDoesNotPolluteHistory(t *testing.T) {
	oldExtrap := demagExtrap
	oldExact := DemagExactEvals
	oldApprox := DemagExtrapolatedEvals
	defer func() {
		demagExtrap = oldExtrap
		DemagExactEvals = oldExact
		DemagExtrapolatedEvals = oldApprox
	}()

	// Output and energy quantities are evaluated only after endStep has made
	// the manager inactive. Their exact-convolution observation must therefore
	// leave both accepted and pending solver history untouched.
	demagExtrap = demagExtrapolator{
		history: []demagHistorySample{{time: 1}, {time: 2}},
		pending: &demagHistorySample{time: 3},
		order:   4,
		active:  false,
	}
	DemagExactEvals = 17
	DemagExtrapolatedEvals = 23

	if tryDemagExtrapolation(nil) {
		t.Fatal("inactive output query took the approximate path")
	}
	observeExactDemag(nil) // models SetDemagField's post-convolution hook

	if len(demagExtrap.history) != 2 || demagExtrap.history[0].time != 1 || demagExtrap.history[1].time != 2 {
		t.Fatalf("exact output query changed accepted history: %#v", demagExtrap.history)
	}
	if demagExtrap.pending == nil || demagExtrap.pending.time != 3 {
		t.Fatalf("exact output query changed pending history: %#v", demagExtrap.pending)
	}
	if DemagExactEvals != 17 || DemagExtrapolatedEvals != 23 {
		t.Fatalf("output query changed solver diagnostics: exact=%d approx=%d", DemagExactEvals, DemagExtrapolatedEvals)
	}
}

func TestRegionwiseRevisionTracksDefinitionsNotStageUpdates(t *testing.T) {
	oldTime := Time
	defer func() { Time = oldTime }()

	var p regionwise
	p.init(1, "test", "", nil)
	p.setFunc(0, 1, func() []float64 { return []float64{Time} })
	revision := p.revision
	if revision == 0 || !p.hasTimeDependence() {
		t.Fatal("time-dependent definition did not update revision/dependency metadata")
	}
	Time = 1
	p.update()
	Time = 2
	p.update()
	if p.revision != revision {
		t.Fatalf("stage-time updates changed definition revision: got %d, want %d", p.revision, revision)
	}
	p.setUniform([]float64{3})
	if p.revision != revision+1 || p.hasTimeDependence() {
		t.Fatal("explicit static redefinition was not tracked correctly")
	}
}
