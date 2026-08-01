package engine

import "testing"

func TestRegionwiseSharedUpdater(t *testing.T) {
	oldTime := Time
	defer func() { Time = oldTime }()

	var p regionwise
	p.init(1, "test", "", nil)
	calls := 0
	p.setFunc(0, NREGION, func() []float64 {
		calls++
		return []float64{Time + 1}
	})

	Time = 2
	got := p.cpuLUT()
	if calls != 1 {
		t.Fatalf("uniform updater called %d times, want 1", calls)
	}
	for r := 0; r < NREGION; r++ {
		if got[0][r] != 3 {
			t.Fatalf("region %d = %v, want 3", r, got[0][r])
		}
	}

	_ = p.cpuLUT()
	if calls != 1 {
		t.Fatalf("updater reevaluated at unchanged time: %d calls", calls)
	}
	Time = 4
	_ = p.cpuLUT()
	if calls != 2 {
		t.Fatalf("updater calls after time change = %d, want 2", calls)
	}
}

func TestRegionwiseDistinctUpdatersAndReplacement(t *testing.T) {
	oldTime := Time
	defer func() { Time = oldTime }()
	Time = 7

	var p regionwise
	p.init(1, "test", "", nil)
	leftCalls, rightCalls := 0, 0
	p.setFunc(0, 128, func() []float64 {
		leftCalls++
		return []float64{1}
	})
	p.setFunc(128, NREGION, func() []float64 {
		rightCalls++
		return []float64{2}
	})
	got := p.cpuLUT()
	if leftCalls != 1 || rightCalls != 1 {
		t.Fatalf("distinct updater calls = (%d, %d), want (1, 1)", leftCalls, rightCalls)
	}
	if got[0][0] != 1 || got[0][127] != 1 || got[0][128] != 2 || got[0][255] != 2 {
		t.Fatalf("unexpected region values around updater boundary")
	}

	// Replacement at the same Time must take effect immediately.
	p.setFunc(0, NREGION, func() []float64 { return []float64{9} })
	got = p.cpuLUT()
	if got[0][0] != 9 || got[0][255] != 9 {
		t.Fatalf("same-time updater replacement was not evaluated")
	}
}

func TestLUTSummaryCacheInvalidation(t *testing.T) {
	var p regionwise
	p.init(2, "test", "", nil)
	p.setRegions(0, NREGION, []float64{0, 0})
	if !p.isZero() || !p.hasZero() || !p.IsUniform() {
		t.Fatalf("zero uniform table summarized incorrectly")
	}

	p.setRegions(17, 18, []float64{1, 2})
	if p.isZero() || !p.hasZero() || p.IsUniform() {
		t.Fatalf("mixed table summarized incorrectly after invalidation")
	}

	p.setRegions(0, NREGION, []float64{3, 4})
	if p.isZero() || p.hasZero() || !p.IsUniform() {
		t.Fatalf("nonzero uniform table summarized incorrectly after invalidation")
	}
}
