package fft

import (
	"math"
	"testing"
)

func TestLayoutMatchesCuFFTRowMajorPacking(t *testing.T) {
	layout, err := NewLayout([]int{3, 4, 5}, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertInts(t, layout.RealShape(), []int{3, 4, 5})
	assertInts(t, layout.HermitianShape(), []int{3, 4, 3})
	assertInts(t, layout.TransformAxes(), []int{0, 1, 2})
	if layout.RealCount() != 60 || layout.HermitianCount() != 36 {
		t.Fatalf("unexpected counts: real=%d Hermitian=%d", layout.RealCount(), layout.HermitianCount())
	}
	if !layout.LastDimensionIsOdd() {
		t.Fatal("last dimension 5 must select odd Hermitian reconstruction")
	}

	batched, err := NewLayout([]int{8}, 7)
	if err != nil {
		t.Fatal(err)
	}
	assertInts(t, batched.RealShape(), []int{7, 8})
	assertInts(t, batched.HermitianShape(), []int{7, 5})
	assertInts(t, batched.TransformAxes(), []int{1})
}

func TestLayoutActivePrefixValidation(t *testing.T) {
	layout, err := NewLayout([]int{1, 128, 256}, 1)
	if err != nil {
		t.Fatal(err)
	}
	hinted := layout.WithActivePrefix(129, 65)
	if hinted.ActiveInner != 129 || hinted.ActiveOuter != 65 {
		t.Fatalf("active prefix = (%d,%d), want (129,65)", hinted.ActiveInner, hinted.ActiveOuter)
	}
	cleared := hinted.WithActivePrefix(256, 128)
	if cleared.ActiveInner != 0 || cleared.ActiveOuter != 0 {
		t.Fatalf("invalid reused prefix retained (%d,%d), want both disabled", cleared.ActiveInner, cleared.ActiveOuter)
	}
	outerCleared := hinted.WithActiveOuter(128)
	if outerCleared.ActiveInner != 129 || outerCleared.ActiveOuter != 0 {
		t.Fatalf("invalid reused outer hint produced (%d,%d), want (129,0)", outerCleared.ActiveInner, outerCleared.ActiveOuter)
	}
	for _, invalid := range [][2]int{{0, 65}, {256, 65}, {129, 0}, {129, 128}} {
		got := layout.WithActivePrefix(invalid[0], invalid[1])
		if invalid[0] <= 0 || invalid[0] >= 256 {
			if got.ActiveInner != 0 {
				t.Fatalf("invalid inner %d enabled hint %d", invalid[0], got.ActiveInner)
			}
		}
		if invalid[1] <= 0 || invalid[1] >= 128 {
			if got.ActiveOuter != 0 {
				t.Fatalf("invalid outer %d enabled hint %d", invalid[1], got.ActiveOuter)
			}
		}
	}
}

func TestReferenceImpulse3D(t *testing.T) {
	dimensions := []int{2, 3, 4}
	input := make([]float32, 24)
	input[0] = 1

	spectrum, err := ReferenceR2C(input, dimensions)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range spectrum {
		if value != 1 {
			t.Fatalf("spectrum[%d]=%v, want 1+0i", i, value)
		}
	}
}

func TestReferenceRoundTripEvenAndOddLastDimensions(t *testing.T) {
	for _, dimensions := range [][]int{{2, 3, 4}, {2, 2, 5}} {
		n := product(dimensions)
		input := make([]float32, n)
		for i := range input {
			input[i] = float32(math.Sin(float64(i)*0.37) + 0.2*math.Cos(float64(i)*0.11))
		}

		spectrum, err := ReferenceR2C(input, dimensions)
		if err != nil {
			t.Fatal(err)
		}
		output, err := ReferenceC2R(spectrum, dimensions)
		if err != nil {
			t.Fatal(err)
		}
		for i := range input {
			want := float32(n) * input[i]
			if diff := math.Abs(float64(output[i] - want)); diff > 2e-4*math.Max(1, math.Abs(float64(want))) {
				t.Fatalf("dims=%v output[%d]=%g, want %g (unnormalised)", dimensions, i, output[i], want)
			}
		}
	}
}

func TestReferenceSingleSinusoidPeak(t *testing.T) {
	const n = 8
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Cos(2 * math.Pi * 2 * float64(i) / n))
	}
	spectrum, err := ReferenceR2C(input, []int{n})
	if err != nil {
		t.Fatal(err)
	}
	for k, value := range spectrum {
		want := complex64(0)
		if k == 2 {
			want = n / 2
		}
		if cmplxAbs(value-want) > 1e-5 {
			t.Fatalf("spectrum[%d]=%v, want %v", k, value, want)
		}
	}
}

func assertInts(t *testing.T, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func cmplxAbs(value complex64) float64 {
	return math.Hypot(float64(real(value)), float64(imag(value)))
}
