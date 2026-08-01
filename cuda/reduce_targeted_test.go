package cuda

import (
	"math"
	"testing"

	"github.com/mumax/3/data"
)

// A targeted reduction waits only its own queue batch and then reads shared
// memory directly instead of draining. It must combine the same partial slots in
// the same order as the ordered path, so the value has to match bit for bit -
// otherwise switching a solver to targeted readback would silently move results.
func TestTargetedReadbackMatchesOrderedReadback(t *testing.T) {
	if !TargetedReadbackSupported {
		t.Skip("targeted readback needs a backend with per-batch completion")
	}
	size := [3]int{1, 64, 128}

	host := data.NewSlice(3, size)
	for c := 0; c < 3; c++ {
		list := host.Host()[c]
		for i := range list {
			// Spread magnitudes across several exponents so that a different
			// addition order would be visible in the last bits.
			list[i] = float32(math.Sin(float64(i+1)*0.37+float64(c)) *
				math.Exp(float64(i%17)-8))
		}
	}

	device := NewSlice(3, size)
	defer device.Free()
	data.Copy(device, host)

	other := NewSlice(3, size)
	defer other.Free()
	Madd2(other, device, device, 0.5, -0.25)

	cases := []struct {
		name           string
		ordered        func() float64
		targeted       func() float64
		requireNonZero bool
	}{
		{
			name:           "MaxVecNorm",
			ordered:        func() float64 { return MaxVecNormAsync(device).Value() },
			targeted:       func() float64 { return MaxVecNormAsync(device).Targeted().Value() },
			requireNonZero: true,
		},
		{
			name:           "Dot",
			ordered:        func() float64 { return DotAsync(device, other).Value() },
			targeted:       func() float64 { return DotAsync(device, other).Targeted().Value() },
			requireNonZero: true,
		},
		{
			name:           "MaxVecDiff",
			ordered:        func() float64 { return MaxVecDiffAsync(device, other).Value() },
			targeted:       func() float64 { return MaxVecDiffAsync(device, other).Targeted().Value() },
			requireNonZero: true,
		},
	}

	for _, test := range cases {
		want := test.ordered()
		got := test.targeted()
		if got != want {
			t.Errorf("%s: targeted readback gave %v, ordered gave %v", test.name, got, want)
		}
		if test.requireNonZero && want == 0 {
			t.Errorf("%s: test data produced 0, which would not detect a difference", test.name)
		}
	}
}

// Resolving a targeted reduction must return its slots to the pool exactly once,
// the same as the ordered path, or a long run starves.
func TestTargetedReadbackRecyclesSlots(t *testing.T) {
	if !TargetedReadbackSupported {
		t.Skip("targeted readback needs a backend with per-batch completion")
	}
	size := [3]int{1, 8, 16}
	device := NewSlice(3, size)
	defer device.Free()
	Memset(device, 1, 2, 3)

	for i := 0; i < 512; i++ {
		pending := MaxVecNormAsync(device).Targeted()
		first := pending.Value()
		// Value is documented idempotent; a second call must not recycle again.
		if second := pending.Value(); second != first {
			t.Fatalf("iteration %d: Value not idempotent: %v then %v", i, first, second)
		}
	}
}
