package cuda

import (
	"fmt"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/cuda/cufft"
	"github.com/mumax/3/data"
	"github.com/mumax/3/timer"
)

// 3D single-precision complex-to-real FFT plan.
type fft3DC2RPlan struct {
	fftplan
	size  [3]int
	batch int
}

// 3D single-precision complex-to-real FFT plan.
// activeY is how many rows along Y can be non-zero, or 0 when all can.
// MuMax3 zero-pads the magnetisation, so the transform along X maps the
// padded rows to zero rows and can skip them exactly.
func newFFT3DC2R(Nx, Ny, Nz, activeY int) fft3DC2RPlan {
	return newFFT3DC2RBatch(Nx, Ny, Nz, 1, activeY)
}

func newFFT3DC2RBatch(Nx, Ny, Nz, batch, activeY int) fft3DC2RPlan {
	handle := cufft.Plan3dPaddedBatch(Nz, Ny, Nx, cufft.C2R, batch, activeY) // new xyz swap
	handle.SetStream(stream0)
	return fft3DC2RPlan{fftplan{handle}, [3]int{Nx, Ny, Nz}, batch}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *fft3DC2RPlan) ExecAsync(src, dst *data.Slice) {
	if Synchronous {
		Sync()
		timer.Start("fft")
	}
	if src.NComp() != p.batch || dst.NComp() != p.batch || !src.Contiguous() || !dst.Contiguous() {
		panic(fmt.Errorf("fft batch mismatch: expecting %d contiguous components, got input=%d output=%d", p.batch, src.NComp(), dst.NComp()))
	}
	oksrclen := p.InputLenFloats()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLenFloats()
	if dst.Len() != okdstlen {
		panic(fmt.Errorf("fft size mismatch: expecting dst len %v, got %v", okdstlen, dst.Len()))
	}
	p.handle.ExecC2R(cu.DevicePtr(uintptr(src.DevPtr(0))), cu.DevicePtr(uintptr(dst.DevPtr(0))))
	if Synchronous {
		Sync()
		timer.Stop("fft")
	}
}

// 3D size of the input array.
func (p *fft3DC2RPlan) InputSizeFloats() (Nx, Ny, Nz int) {
	return 2 * (p.size[X]/2 + 1), p.size[Y], p.size[Z]
}

// 3D size of the output array.
func (p *fft3DC2RPlan) OutputSizeFloats() (Nx, Ny, Nz int) {
	return p.size[X], p.size[Y], p.size[Z]
}

// Required length of the (1D) input array.
func (p *fft3DC2RPlan) InputLenFloats() int {
	return prod3(p.InputSizeFloats())
}

// Required length of the (1D) output array.
func (p *fft3DC2RPlan) OutputLenFloats() int {
	return prod3(p.OutputSizeFloats())
}
