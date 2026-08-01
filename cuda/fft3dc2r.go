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
	size    [3]int
	inPlace bool
}

// 3D single-precision complex-to-real FFT plan.
// activeX/activeY describe the strict non-zero prefix used by the optional
// Metal in-place backend. Pass zero to retain the regular out-of-place plan.
func newFFT3DC2R(Nx, Ny, Nz, activeX, activeY int) fft3DC2RPlan {
	handle := cufft.Plan3dPadded(Nz, Ny, Nx, cufft.C2R, activeX, activeY) // new xyz swap
	handle.SetStream(stream0)
	return fft3DC2RPlan{fftplan{handle}, [3]int{Nx, Ny, Nz}, handle.InPlace()}
}

// Execute the FFT plan, asynchronous.
// src and dst are 3D arrays stored 1D arrays.
func (p *fft3DC2RPlan) ExecAsync(src, dst *data.Slice) {
	if Synchronous {
		Sync()
		timer.Start("fft")
	}
	oksrclen := p.InputLenFloats()
	if src.Len() != oksrclen {
		panic(fmt.Errorf("fft size mismatch: expecting src len %v, got %v", oksrclen, src.Len()))
	}
	okdstlen := p.OutputLenFloats()
	if p.inPlace {
		okdstlen = p.InputLenFloats()
	}
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
