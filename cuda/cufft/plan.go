//go:build !darwin || !arm64
// +build !darwin !arm64

// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package cufft

//#include <cufft.h>
import "C"

import (
	"unsafe"

	"github.com/mumax/3/cuda/cu"
)

// FFT plan handle, reference type to a plan
type Handle uintptr

// 1D FFT plan
func Plan1d(nx int, typ Type, batch int) Handle {
	var handle C.cufftHandle
	err := Result(C.cufftPlan1d(
		&handle,
		C.int(nx),
		C.cufftType(typ),
		C.int(batch)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(handle)
}

// 2D FFT plan
func Plan2d(nx, ny int, typ Type) Handle {
	var handle C.cufftHandle
	err := Result(C.cufftPlan2d(
		&handle,
		C.int(nx),
		C.int(ny),
		C.cufftType(typ)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(handle)
}

// 3D FFT plan
func Plan3d(nx, ny, nz int, typ Type) Handle {
	var handle C.cufftHandle
	err := Result(C.cufftPlan3d(
		&handle,
		C.int(nx),
		C.int(ny),
		C.int(nz),
		C.cufftType(typ)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(handle)
}

//cufftPlanMany(
//    cufftHandle *plan, int rank, int *n, int *inembed,
//    int istride, int idist, int *onembed, int ostride,
//    int odist, cufftType type, int batch );

// 1D,2D or 3D FFT plan
// Plan3dPadded is Plan3d for an array whose trailing rows along the
// second-to-last axis are known to be zero. cuFFT has no way to express that,
// so activeOuter is ignored here; the Metal backend uses it to skip the zero
// rows of MuMax3's zero-padded arrays.
func Plan3dPadded(nx, ny, nz int, typ Type, activeOuter int) Handle {
	_ = activeOuter
	return Plan3d(nx, ny, nz, typ)
}

// Plan3dPaddedBatch creates tightly packed, contiguous 3D transforms. cuFFT
// infers the input and output distances when the embed arrays are nil.
func Plan3dPaddedBatch(nx, ny, nz int, typ Type, batch, activeOuter int) Handle {
	_ = activeOuter
	if batch == 1 {
		return Plan3d(nx, ny, nz, typ)
	}
	return PlanMany([]int{nx, ny, nz}, nil, 1, nil, 1, typ, batch)
}

func PlanMany(n []int, inembed []int, istride int, oembed []int, ostride int, typ Type, batch int) Handle {
	var handle C.cufftHandle

	NULL := (*C.int)(unsafe.Pointer(uintptr(0)))

	inembedptr := NULL
	idist := 0
	if inembed != nil {
		inembedptr = (*C.int)(unsafe.Pointer(&inembed[0]))
		idist = inembed[0]
	}

	oembedptr := NULL
	odist := 0
	if oembed != nil {
		oembedptr = (*C.int)(unsafe.Pointer(&oembed[0]))
		odist = oembed[0]
	}

	err := Result(C.cufftPlanMany(
		&handle,
		C.int(len(n)),                   // rank
		(*C.int)(unsafe.Pointer(&n[0])), // n
		inembedptr,
		C.int(istride),
		C.int(idist),
		oembedptr,
		C.int(ostride),
		C.int(odist),
		C.cufftType(typ),
		C.int(batch)))
	if err != SUCCESS {
		panic(err)
	}
	return Handle(handle)
}

// Execute Complex-to-Complex plan
func (plan Handle) ExecC2C(idata, odata cu.DevicePtr, direction int) {
	err := Result(C.cufftExecC2C(
		C.cufftHandle(plan),
		(*C.cufftComplex)(unsafe.Pointer(uintptr(idata))),
		(*C.cufftComplex)(unsafe.Pointer(uintptr(odata))),
		C.int(direction)))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Real-to-Complex plan
func (plan Handle) ExecR2C(idata, odata cu.DevicePtr) {
	err := Result(C.cufftExecR2C(
		C.cufftHandle(plan),
		(*C.cufftReal)(unsafe.Pointer(uintptr(idata))),
		(*C.cufftComplex)(unsafe.Pointer(uintptr(odata)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Complex-to-Real plan
func (plan Handle) ExecC2R(idata, odata cu.DevicePtr) {
	err := Result(C.cufftExecC2R(
		C.cufftHandle(plan),
		(*C.cufftComplex)(unsafe.Pointer(uintptr(idata))),
		(*C.cufftReal)(unsafe.Pointer(uintptr(odata)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Double Complex-to-Complex plan
func (plan Handle) ExecZ2Z(idata, odata cu.DevicePtr, direction int) {
	err := Result(C.cufftExecZ2Z(
		C.cufftHandle(plan),
		(*C.cufftDoubleComplex)(unsafe.Pointer(uintptr(idata))),
		(*C.cufftDoubleComplex)(unsafe.Pointer(uintptr(odata))),
		C.int(direction)))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Double Real-to-Complex plan
func (plan Handle) ExecD2Z(idata, odata cu.DevicePtr) {
	err := Result(C.cufftExecD2Z(
		C.cufftHandle(plan),
		(*C.cufftDoubleReal)(unsafe.Pointer(uintptr(idata))),
		(*C.cufftDoubleComplex)(unsafe.Pointer(uintptr(odata)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Execute Double Complex-to-Real plan
func (plan Handle) ExecZ2D(idata, odata cu.DevicePtr) {
	err := Result(C.cufftExecZ2D(
		C.cufftHandle(plan),
		(*C.cufftDoubleComplex)(unsafe.Pointer(uintptr(idata))),
		(*C.cufftDoubleReal)(unsafe.Pointer(uintptr(odata)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Destroys the plan.
func (plan *Handle) Destroy() {
	err := Result(C.cufftDestroy(C.cufftHandle(*plan)))
	*plan = 0 // make sure plan is not used anymore
	if err != SUCCESS {
		panic(err)
	}
}

// Sets the cuda stream for this plan
func (plan Handle) SetStream(stream cu.Stream) {
	err := Result(C.cufftSetStream(
		C.cufftHandle(plan),
		C.cudaStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}
