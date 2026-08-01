//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

package cuda

import (
	"math"
	"testing"

	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
)

func TestDemagConvolutionVkFFTMatchesMPS(t *testing.T) {
	inputSize := [3]int{8, 4, 1}
	pbc := [3]int{}
	hostKernel := mag.CalcDemagKernel(inputSize, pbc, [3]float64{3e-9, 4e-9, 5e-9}, 4)
	defer func() {
		for i := 0; i < 3; i++ {
			for j := i; j < 3; j++ {
				hostKernel[i][j].Free()
			}
		}
	}()

	var deviceKernel [3][3]*data.Slice
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			if hostKernel[i][j] == nil {
				continue
			}
			deviceKernel[i][j] = NewSlice(1, hostKernel[i][j].Size())
			data.Copy(deviceKernel[i][j], hostKernel[i][j])
			defer deviceKernel[i][j].Free()
		}
	}

	hostInput := data.NewSlice(3, inputSize)
	defer hostInput.Free()
	input := NewSlice(3, inputSize)
	defer input.Free()

	msat := NewSlice(1, inputSize)
	defer msat.Free()
	Memset(msat, 800e3)
	volume := data.NilSlice(1, inputSize)

	t.Setenv("MUMAX3_METAL_FFT_BACKEND", "mps")
	mps := NewDemag(inputSize, pbc, deviceKernel, false)
	defer mps.Free()
	if mps.fftInPlace {
		t.Fatal("MPS reference convolution unexpectedly selected an in-place plan")
	}

	t.Setenv("MUMAX3_METAL_FFT_BACKEND", "vkfft")
	vk := NewDemag(inputSize, pbc, deviceKernel, false)
	defer vk.Free()
	if !vk.fftInPlace {
		t.Fatal("forced eligible convolution did not select VkFFT")
	}

	mpsOutput := NewSlice(3, inputSize)
	vkOutput := NewSlice(3, inputSize)
	defer mpsOutput.Free()
	defer vkOutput.Free()
	for iteration := 0; iteration < 3; iteration++ {
		// The first in-place inverse overwrites the padded Y rows. Changing the
		// input and running again exercises the next forward transform's narrow
		// Y-tail clear instead of relying on fresh zero-filled allocations.
		for component, values := range hostInput.Host() {
			for index := range values {
				x := index % inputSize[X]
				y := index / inputSize[X]
				values[index] = float32(
					0.37*math.Sin(float64((component+2+iteration)*x+3*y)*0.29) +
						0.19*math.Cos(float64(5*x-(component+1)*y+iteration)*0.17) +
						0.03*float64(component+1+iteration),
				)
			}
		}
		data.Copy(input, hostInput)
		mps.Exec(mpsOutput, input, volume, ToMSlice(msat))
		vk.Exec(vkOutput, input, volume, ToMSlice(msat))
		Sync()

		want := mpsOutput.HostCopy()
		got := vkOutput.HostCopy()
		wantValues := want.Host()
		gotValues := got.Host()
		maxDifference := 0.0
		maxMagnitude := 0.0
		for component := range wantValues {
			for index, reference := range wantValues[component] {
				referenceValue := float64(reference)
				candidate := float64(gotValues[component][index])
				if math.IsNaN(referenceValue) || math.IsInf(referenceValue, 0) ||
					math.IsNaN(candidate) || math.IsInf(candidate, 0) {
					t.Fatalf("iteration %d: non-finite demag output at component %d index %d: MPS=%g, VkFFT=%g", iteration, component, index, referenceValue, candidate)
				}
				magnitude := math.Abs(referenceValue)
				if magnitude > maxMagnitude {
					maxMagnitude = magnitude
				}
				difference := math.Abs(candidate - referenceValue)
				if difference > maxDifference {
					maxDifference = difference
				}
			}
		}
		want.Free()
		got.Free()
		tolerance := 2e-6 * math.Max(1, maxMagnitude)
		if maxDifference > tolerance {
			t.Fatalf("iteration %d: VkFFT demag differs from MPS: max |ΔB|=%g T, tolerance=%g T (peak=%g T)", iteration, maxDifference, tolerance, maxMagnitude)
		}
	}
}
