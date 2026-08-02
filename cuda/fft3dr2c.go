package cuda

import (
	"log"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/cuda/cufft"
	"github.com/mumax/3/data"
	"github.com/mumax/3/timer"
	"github.com/mumax/3/util"
)

// 3D single-precision real-to-complex FFT plan.
type fft3DR2CPlan struct {
	fftplan
	size  [3]int
	batch int
}

// 3D single-precision real-to-complex FFT plan.
// activeY is how many rows along Y can be non-zero, or 0 when all can.
// MuMax3 zero-pads the magnetisation, so the transform along X maps the
// padded rows to zero rows and can skip them exactly.
func newFFT3DR2C(Nx, Ny, Nz, activeY int) fft3DR2CPlan {
	return newFFT3DR2CBatch(Nx, Ny, Nz, 1, activeY)
}

// newFFT3DR2CBatch creates a plan for tightly packed component-major arrays.
func newFFT3DR2CBatch(Nx, Ny, Nz, batch, activeY int) fft3DR2CPlan {
	handle := cufft.Plan3dPaddedBatch(Nz, Ny, Nx, cufft.R2C, batch, activeY) // new xyz swap
	handle.SetStream(stream0)
	return fft3DR2CPlan{fftplan{handle}, [3]int{Nx, Ny, Nz}, batch}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *fft3DR2CPlan) ExecAsync(src, dst *data.Slice) {
	if Synchronous {
		Sync()
		timer.Start("fft")
	}
	util.Argument(src.NComp() == p.batch && dst.NComp() == p.batch)
	util.Argument(src.Contiguous() && dst.Contiguous())
	oksrclen := p.InputLen()
	if src.Len() != oksrclen {
		log.Panicf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len())
	}
	okdstlen := p.OutputLen()
	if dst.Len() != okdstlen {
		log.Panicf("fft size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len())
	}
	p.handle.ExecR2C(cu.DevicePtr(uintptr(src.DevPtr(0))), cu.DevicePtr(uintptr(dst.DevPtr(0))))
	if Synchronous {
		Sync()
		timer.Stop("fft")
	}
}

// 3D size of the input array.
func (p *fft3DR2CPlan) InputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[X], p.size[Y], p.size[Z]
}

// 3D size of the output array.
func (p *fft3DR2CPlan) OutputSizeFloats() (Nx, Ny, Nz int) {
	return 2 * (p.size[X]/2 + 1), p.size[Y], p.size[Z]
}

// Required length of the (1D) input array.
func (p *fft3DR2CPlan) InputLen() int {
	return prod3(p.InputSizeFloats())
}

// Required length of the (1D) output array.
func (p *fft3DR2CPlan) OutputLen() int {
	return prod3(p.OutputSizeFloats())
}
