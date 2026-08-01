//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

package fft

import (
	"fmt"
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/mumax/3/cuda/metal"
)

func TestExternalCommitAndContinueHandoff(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}

	const bytes = int64(4096)
	buffer := mustMetalAlloc(t, bytes)
	defer mustMetalFree(t, buffer)

	// Exceed the runtime's retained-command-buffer bound while repeatedly
	// replacing the root. The final fill must execute after every committed
	// predecessor, and Sync must commit only the last live root.
	for value := byte(1); value <= 12; value++ {
		if err := metal.Fill(buffer, value, bytes); err != nil {
			t.Fatalf("fill before handoff %d: %v", value, err)
		}
		if err := forceCommitAndContinueForTest(); err != nil {
			t.Fatalf("handoff %d: %v", value, err)
		}
	}
	const finalValue = byte(0x7f)
	if err := metal.Fill(buffer, finalValue, bytes); err != nil {
		t.Fatal(err)
	}
	host := make([]byte, bytes)
	if err := metal.CopyToHost(
		unsafe.Pointer(unsafe.SliceData(host)),
		buffer,
		bytes,
	); err != nil {
		t.Fatal(err)
	}
	for index, value := range host {
		if value != finalValue {
			t.Fatalf("result[%d] = %#x, want %#x", index, value, finalValue)
		}
	}
}

func TestCompletionSurvivesMPSRootAdoption(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	const bytes = int64(4096)
	buffer := mustMetalAlloc(t, bytes)
	defer mustMetalFree(t, buffer)
	if err := metal.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := metal.ResetRuntimeStats(); err != nil {
		t.Fatal(err)
	}

	if err := metal.Fill(buffer, 3, bytes); err != nil {
		t.Fatal(err)
	}
	completion, err := metal.RecordCompletion()
	if err != nil {
		t.Fatal(err)
	}
	if !completion.Valid() {
		t.Fatal("recording the pre-MPS fill returned no completion")
	}
	defer completion.Close()
	if err := forceCommitAndContinueForTest(); err != nil {
		t.Fatal(err)
	}
	if err := completion.Wait(); err != nil {
		t.Fatal(err)
	}

	stats, err := metal.GetRuntimeStats()
	if err != nil {
		t.Fatal(err)
	}
	if stats.ExternalRootAdoptions != 1 {
		t.Fatalf("root adoptions = %d, want 1", stats.ExternalRootAdoptions)
	}
	if stats.CompletionWaitSubmissions != 0 {
		t.Fatalf(
			"waiting an adopted old root submitted %d command buffers, want 0",
			stats.CompletionWaitSubmissions,
		)
	}
	if stats.FullDrains != 0 {
		t.Fatalf("completion wait performed %d full drains", stats.FullDrains)
	}

	if err := metal.Fill(buffer, 4, bytes); err != nil {
		t.Fatal(err)
	}
	host := make([]byte, bytes)
	if err := metal.CopyToHost(
		unsafe.Pointer(unsafe.SliceData(host)),
		buffer,
		bytes,
	); err != nil {
		t.Fatal(err)
	}
	for index, value := range host {
		if value != 4 {
			t.Fatalf("result[%d] = %d, want 4", index, value)
		}
	}
}

func TestMetalFFT3DRoundTripEvenAndOdd(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	for _, dimensions := range [][]int{{2, 3, 4}, {2, 3, 5}} {
		t.Run(shapeName(dimensions), func(t *testing.T) {
			layout, err := NewLayout(dimensions, 1)
			if err != nil {
				t.Fatal(err)
			}
			input := make([]float32, layout.RealCount())
			for i := range input {
				input[i] = float32(math.Sin(float64(i)*0.29) + 0.25*math.Cos(float64(i)*0.17))
			}

			realBuffer := mustMetalAlloc(t, int64(layout.RealCount()*4))
			spectrumBuffer := mustMetalAlloc(t, int64(layout.HermitianCount()*8))
			roundTripBuffer := mustMetalAlloc(t, int64(layout.RealCount()*4))
			defer mustMetalFree(t, realBuffer)
			defer mustMetalFree(t, spectrumBuffer)
			defer mustMetalFree(t, roundTripBuffer)
			if err := metal.CopyToDevice(realBuffer, unsafe.Pointer(&input[0]), int64(len(input)*4)); err != nil {
				t.Fatal(err)
			}

			forward, err := CreatePlan(layout, RealToComplex)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := DestroyPlan(forward); err != nil {
					t.Error(err)
				}
			}()
			inverse, err := CreatePlan(layout, ComplexToReal)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := DestroyPlan(inverse); err != nil {
					t.Error(err)
				}
			}()

			if err := Execute(forward, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
				t.Fatal(err)
			}
			if err := Execute(inverse, uintptr(spectrumBuffer), uintptr(roundTripBuffer), 1); err != nil {
				t.Fatal(err)
			}
			if err := metal.Sync(); err != nil {
				t.Fatal(err)
			}

			output := make([]float32, len(input))
			if err := metal.CopyToHost(unsafe.Pointer(&output[0]), roundTripBuffer, int64(len(output)*4)); err != nil {
				t.Fatal(err)
			}
			scale := float32(layout.RealCount())
			for i := range input {
				want := scale * input[i]
				tolerance := 8e-4 * math.Max(1, math.Abs(float64(want)))
				if difference := math.Abs(float64(output[i] - want)); nonFinite(difference) || difference > tolerance {
					t.Fatalf("dims=%v output[%d]=%g, want %g ± %g", dimensions, i, output[i], want, tolerance)
				}
			}
		})
	}
}

func TestMetalFFT3DBatchedRoundTrip(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	const batch = 3
	layout, err := NewLayout([]int{2, 3, 5}, batch)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, layout.RealCount())
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.19) + 0.03*float64(i))
	}

	realBuffer := mustMetalAlloc(t, int64(layout.RealCount()*4))
	spectrumBuffer := mustMetalAlloc(t, int64(layout.HermitianCount()*8))
	roundTripBuffer := mustMetalAlloc(t, int64(layout.RealCount()*4))
	defer mustMetalFree(t, realBuffer)
	defer mustMetalFree(t, spectrumBuffer)
	defer mustMetalFree(t, roundTripBuffer)
	if err := metal.CopyToDevice(
		realBuffer,
		unsafe.Pointer(unsafe.SliceData(input)),
		int64(len(input)*4),
	); err != nil {
		t.Fatal(err)
	}

	forward, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(forward); err != nil {
			t.Error(err)
		}
	}()
	inverse, err := CreatePlan(layout, ComplexToReal)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(inverse); err != nil {
			t.Error(err)
		}
	}()

	if err := Execute(forward, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
		t.Fatal(err)
	}
	if err := Execute(inverse, uintptr(spectrumBuffer), uintptr(roundTripBuffer), 1); err != nil {
		t.Fatal(err)
	}
	output := make([]float32, len(input))
	if err := metal.CopyToHost(
		unsafe.Pointer(unsafe.SliceData(output)),
		roundTripBuffer,
		int64(len(output)*4),
	); err != nil {
		t.Fatal(err)
	}

	scale := float32(product(layout.Dimensions))
	for i := range input {
		want := scale * input[i]
		tolerance := 8e-4 * math.Max(1, math.Abs(float64(want)))
		if difference := math.Abs(float64(output[i] - want)); nonFinite(difference) || difference > tolerance {
			t.Fatalf("output[%d]=%g, want %g ± %g", i, output[i], want, tolerance)
		}
	}
}

func TestMetalFFTLargePadded2DSmoke(t *testing.T) {
	if os.Getenv("MUMAX3_METAL_LARGE_FFT_TEST") != "1" {
		t.Skip("set MUMAX3_METAL_LARGE_FFT_TEST=1 to exercise the padded 8192² FFT")
	}
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	layout, err := NewLayout([]int{1, 8192, 8192}, 1)
	if err != nil {
		t.Fatal(err)
	}
	layout = layout.WithActiveOuter(4096)
	realBuffer := mustMetalAlloc(t, int64(layout.RealCount()*4))
	spectrumBuffer := mustMetalAlloc(t, int64(layout.HermitianCount()*8))
	defer mustMetalFree(t, realBuffer)
	defer mustMetalFree(t, spectrumBuffer)
	if err := metal.Fill(realBuffer, 0, int64(layout.RealCount()*4)); err != nil {
		t.Fatal(err)
	}
	plan, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(plan); err != nil {
			t.Error(err)
		}
	}()
	if err := Execute(plan, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
		t.Fatal(err)
	}
	var first complex64
	if err := metal.CopyToHost(
		unsafe.Pointer(&first),
		spectrumBuffer,
		int64(unsafe.Sizeof(first)),
	); err != nil {
		t.Fatal(err)
	}
	if first != 0 {
		t.Fatalf("zero-input DC term = %v, want 0", first)
	}
}

func TestMetalFFTR2CImpulsePacking(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	layout, err := NewLayout([]int{3, 4, 5}, 1)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, layout.RealCount())
	input[0] = 1
	spectrum := make([]complex64, layout.HermitianCount())

	realBuffer := mustMetalAlloc(t, int64(len(input)*4))
	spectrumBuffer := mustMetalAlloc(t, int64(len(spectrum)*8))
	defer mustMetalFree(t, realBuffer)
	defer mustMetalFree(t, spectrumBuffer)
	if err := metal.CopyToDevice(realBuffer, unsafe.Pointer(&input[0]), int64(len(input)*4)); err != nil {
		t.Fatal(err)
	}
	plan, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(plan); err != nil {
			t.Error(err)
		}
	}()
	if err := Execute(plan, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
		t.Fatal(err)
	}
	if err := metal.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := metal.CopyToHost(unsafe.Pointer(&spectrum[0]), spectrumBuffer, int64(len(spectrum)*8)); err != nil {
		t.Fatal(err)
	}
	for i, value := range spectrum {
		if difference := cmplxAbs(value - 1); nonFinite(difference) || difference > 2e-5 {
			t.Fatalf("packed spectrum[%d]=%v, want 1+0i", i, value)
		}
	}
}

func TestMetalFFTR2CMatchesReferenceSpectrum(t *testing.T) {
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	layout, err := NewLayout([]int{2, 3, 5}, 1)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, layout.RealCount())
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*0.41) + 0.07*float64(i))
	}
	want, err := ReferenceR2C(input, layout.Dimensions)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]complex64, layout.HermitianCount())

	realBuffer := mustMetalAlloc(t, int64(len(input)*4))
	spectrumBuffer := mustMetalAlloc(t, int64(len(got)*8))
	defer mustMetalFree(t, realBuffer)
	defer mustMetalFree(t, spectrumBuffer)
	if err := metal.CopyToDevice(realBuffer, unsafe.Pointer(&input[0]), int64(len(input)*4)); err != nil {
		t.Fatal(err)
	}
	plan, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(plan); err != nil {
			t.Error(err)
		}
	}()
	if err := Execute(plan, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
		t.Fatal(err)
	}
	if err := metal.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := metal.CopyToHost(unsafe.Pointer(&got[0]), spectrumBuffer, int64(len(got)*8)); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		tolerance := 5e-4 * math.Max(1, cmplxAbs(want[i]))
		if difference := cmplxAbs(got[i] - want[i]); nonFinite(difference) || difference > tolerance {
			t.Fatalf("packed spectrum[%d]=%v, want %v ± %g", i, got[i], want[i], tolerance)
		}
	}
}

func TestVkFFTInPlacePaddedRoundTrip(t *testing.T) {
	t.Setenv("MUMAX3_METAL_FFT_BACKEND", "vkfft")
	if err := metal.Initialize(); err != nil {
		t.Fatal(err)
	}
	const (
		nx      = 8
		ny      = 4
		activeX = 3
		activeY = 2
		stride  = nx + 2
	)
	layout, err := NewLayout([]int{1, ny, nx}, 1)
	if err != nil {
		t.Fatal(err)
	}
	layout = layout.WithActivePrefix(activeX, activeY)
	buffer := mustMetalAlloc(t, int64((nx+2)*ny*4))
	defer mustMetalFree(t, buffer)

	input := make([]float32, stride*ny)
	logicalInput := make([]float32, nx*ny)
	for y := 0; y < activeY; y++ {
		for x := 0; x < activeX; x++ {
			value := float32(math.Sin(float64(3*x+5*y)*0.31) + 0.07*float64(x-y))
			input[y*stride+x] = value
			logicalInput[y*nx+x] = value
		}
	}
	if err := metal.CopyToDevice(buffer, unsafe.Pointer(unsafe.SliceData(input)), int64(len(input)*4)); err != nil {
		t.Fatal(err)
	}

	forward, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(forward); err != nil {
			t.Error(err)
		}
	}()
	inverse, err := CreatePlan(layout, ComplexToReal)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(inverse); err != nil {
			t.Error(err)
		}
	}()
	if !PlanIsInPlace(forward) || !PlanIsInPlace(inverse) {
		t.Fatal("forced eligible VkFFT plans did not select the in-place backend")
	}

	for iteration := 0; iteration < 32; iteration++ {
		if err := Execute(forward, uintptr(buffer), uintptr(buffer), -1); err != nil {
			t.Fatalf("forward %d: %v", iteration, err)
		}
		if iteration == 0 {
			spectrum := make([]complex64, (nx/2+1)*ny)
			if err := metal.CopyToHost(unsafe.Pointer(unsafe.SliceData(spectrum)), buffer, int64(len(spectrum)*8)); err != nil {
				t.Fatal(err)
			}
			wantSpectrum, err := ReferenceR2C(logicalInput, []int{1, ny, nx})
			if err != nil {
				t.Fatal(err)
			}
			for index := range wantSpectrum {
				tolerance := 5e-4 * math.Max(1, cmplxAbs(wantSpectrum[index]))
				if difference := cmplxAbs(spectrum[index] - wantSpectrum[index]); nonFinite(difference) || difference > tolerance {
					t.Fatalf("forward spectrum[%d]=%v, want %v ± %g", index, spectrum[index], wantSpectrum[index], tolerance)
				}
			}
		}
		if err := Execute(inverse, uintptr(buffer), uintptr(buffer), 1); err != nil {
			t.Fatalf("inverse %d: %v", iteration, err)
		}
		if iteration != 31 {
			// An unnormalized round trip scales by nx*ny, so restore the known
			// input between repetitions while retaining the encoder stress.
			if err := metal.CopyToDevice(buffer, unsafe.Pointer(unsafe.SliceData(input)), int64(len(input)*4)); err != nil {
				t.Fatal(err)
			}
		}
	}
	output := make([]float32, len(input))
	if err := metal.CopyToHost(unsafe.Pointer(unsafe.SliceData(output)), buffer, int64(len(output)*4)); err != nil {
		t.Fatal(err)
	}
	scale := float32(nx * ny)
	for y := 0; y < activeY; y++ {
		for x := 0; x < activeX; x++ {
			index := y*stride + x
			want := scale * input[index]
			tolerance := 8e-4 * math.Max(1, math.Abs(float64(want)))
			if difference := math.Abs(float64(output[index] - want)); nonFinite(difference) || difference > tolerance {
				t.Fatalf("output[%d,%d]=%g, want %g ± %g", x, y, output[index], want, tolerance)
			}
		}
	}
}

func TestVkFFTEligibilityGate(t *testing.T) {
	t.Setenv("MUMAX3_METAL_FFT_BACKEND", "auto")
	makeLayout := func(dimensions []int, activeX, activeY int) Layout {
		t.Helper()
		layout, err := NewLayout(dimensions, 1)
		if err != nil {
			t.Fatal(err)
		}
		return layout.WithActivePrefix(activeX, activeY)
	}
	tests := []struct {
		name      string
		layout    Layout
		transform Transform
		want      bool
	}{
		{"automatic-window", makeLayout([]int{1, 512, 512}, 256, 256), RealToComplex, true},
		{"padded-1024-is-opt-in", makeLayout([]int{1, 1024, 1024}, 512, 512), RealToComplex, false},
		{"real-1024-pads-to-2048", makeLayout([]int{1, 2048, 2048}, 1024, 1024), RealToComplex, false},
		{"odd-logical-fast-axis", makeLayout([]int{1, 512, 999}, 499, 256), RealToComplex, false},
		{"non-power-of-two-fast-axis", makeLayout([]int{1, 512, 1006}, 503, 256), RealToComplex, false},
		{"non-power-of-two-outer-axis", makeLayout([]int{1, 768, 512}, 256, 384), RealToComplex, false},
		{"no-strict-inner-padding", makeLayout([]int{1, 512, 512}, 512, 256), RealToComplex, false},
		{"complex-transform", makeLayout([]int{1, 512, 512}, 256, 256), ComplexToComplex, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := vkFFTEligible(test.layout, test.transform); got != test.want {
				t.Fatalf("vkFFTEligible(%v, %d) = %v, want %v", test.layout, test.transform, got, test.want)
			}
		})
	}

	t.Setenv("MUMAX3_METAL_FFT_BACKEND", "mps")
	if vkFFTEligible(makeLayout([]int{1, 512, 512}, 256, 256), RealToComplex) {
		t.Fatal("MPS override must disable VkFFT")
	}

	t.Setenv("MUMAX3_METAL_FFT_BACKEND", "vkfft")
	if !vkFFTEligible(makeLayout([]int{1, 1024, 1024}, 512, 512), RealToComplex) {
		t.Fatal("forced VkFFT must permit the measured padded-1024 tier")
	}
}

func BenchmarkMetalFFT3DRoundTrip(b *testing.B) {
	if err := metal.Initialize(); err != nil {
		b.Fatal(err)
	}
	layout, err := NewLayout([]int{32, 128, 128}, 1)
	if err != nil {
		b.Fatal(err)
	}
	realBuffer, err := metal.Alloc(int64(layout.RealCount() * 4))
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := metal.Free(realBuffer); err != nil {
			b.Error(err)
		}
	}()
	spectrumBuffer, err := metal.Alloc(int64(layout.HermitianCount() * 8))
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := metal.Free(spectrumBuffer); err != nil {
			b.Error(err)
		}
	}()
	outputBuffer, err := metal.Alloc(int64(layout.RealCount() * 4))
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := metal.Free(outputBuffer); err != nil {
			b.Error(err)
		}
	}()
	forward, err := CreatePlan(layout, RealToComplex)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(forward); err != nil {
			b.Error(err)
		}
	}()
	inverse, err := CreatePlan(layout, ComplexToReal)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := DestroyPlan(inverse); err != nil {
			b.Error(err)
		}
	}()

	if err := Execute(forward, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
		b.Fatal(err)
	}
	if err := Execute(inverse, uintptr(spectrumBuffer), uintptr(outputBuffer), 1); err != nil {
		b.Fatal(err)
	}
	if err := metal.Sync(); err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(layout.RealCount() * 4 * 2))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Execute(forward, uintptr(realBuffer), uintptr(spectrumBuffer), -1); err != nil {
			b.Fatal(err)
		}
		if err := Execute(inverse, uintptr(spectrumBuffer), uintptr(outputBuffer), 1); err != nil {
			b.Fatal(err)
		}
		if err := metal.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func mustMetalAlloc(t *testing.T, bytes int64) unsafe.Pointer {
	t.Helper()
	pointer, err := metal.Alloc(bytes)
	if err != nil {
		t.Fatal(err)
	}
	return pointer
}

func mustMetalFree(t *testing.T, pointer unsafe.Pointer) {
	t.Helper()
	if err := metal.Free(pointer); err != nil {
		t.Error(err)
	}
}

func nonFinite(value float64) bool {
	return math.IsNaN(value) || math.IsInf(value, 0)
}

func shapeName(dimensions []int) string {
	result := ""
	for i, dimension := range dimensions {
		if i != 0 {
			result += "x"
		}
		result += fmt.Sprint(dimension)
	}
	return result
}
