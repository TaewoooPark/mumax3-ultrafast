//go:build darwin && arm64
// +build darwin,arm64

package cuda

import (
	"log"
	"math"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// compressFFTKernel keeps the full FFT spectrum on the GPU. Only the scalar
// maximum imaginary residual crosses to the CPU for the existing sanity check.
func compressFFTKernel(src *data.Slice, dstSize [3]int, scale float32) *data.Slice {
	util.Argument(src.NComp() == 1)
	srcSize := src.Size()
	util.Argument(dstSize[X]*2 == srcSize[X])
	util.Argument(dstSize[Y] == srcSize[Y]/2+1)
	util.Argument(dstSize[Z] == srcSize[Z]/2+1)

	dst := NewSlice(1, dstSize)
	imag := NewSlice(1, dstSize)
	defer imag.Free()
	k_compresskernel_async(
		dst.DevPtr(0), imag.DevPtr(0),
		dstSize[X], dstSize[Y], dstSize[Z],
		src.DevPtr(0), srcSize[X], srcSize[Y], srcSize[Z],
		scale, float32(math.Sqrt(float64(scale))), make3DConf(dstSize),
	)
	if maxImag := MaxAbs(imag); maxImag > FFT_IMAG_TOLERANCE {
		dst.Free()
		log.Fatalf("FFT kernel imaginary part: %v\n", maxImag)
	}
	return dst
}
