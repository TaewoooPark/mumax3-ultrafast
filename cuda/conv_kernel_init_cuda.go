//go:build !darwin || !arm64
// +build !darwin !arm64

package cuda

import "github.com/mumax/3/data"

// compressFFTKernel retains the established host implementation on CUDA. The
// Metal implementation avoids these full-spectrum device/host transfers.
func compressFFTKernel(src *data.Slice, dstSize [3]int, scale float32) *data.Slice {
	kfull := data.NewSlice(1, src.Size())
	defer kfull.Free()
	data.Copy(kfull, src)

	complexSize := dstSize
	complexSize[X] *= 2
	kComplex := data.NewSlice(1, complexSize)
	defer kComplex.Free()
	full := kfull.Scalars()
	compressed := kComplex.Scalars()
	for iz := 0; iz < complexSize[Z]; iz++ {
		for iy := 0; iy < complexSize[Y]; iy++ {
			for ix := 0; ix < complexSize[X]; ix++ {
				compressed[iz][iy][ix] = full[iz][iy][ix]
			}
		}
	}

	hostReal := data.NewSlice(1, dstSize)
	defer hostReal.Free()
	scaleRealParts(hostReal, kComplex, scale)
	return GPUCopy(hostReal)
}
