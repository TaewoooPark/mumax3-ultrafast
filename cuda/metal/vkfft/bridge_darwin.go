//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

// Package vkfft is the non-ARC Metal bridge for the vendored VkFFT backend.
package vkfft

/*
#cgo darwin,arm64 CFLAGS: -mmacosx-version-min=14.0
#cgo darwin,arm64 CXXFLAGS: -std=c++17 -fno-objc-arc -Wno-deprecated-declarations -Wno-comment -mmacosx-version-min=14.0 -I${SRCDIR}/third_party/vkfft/metal-cpp -I${SRCDIR}/third_party/vkfft/vkFFT
#cgo darwin,arm64 LDFLAGS: -mmacosx-version-min=14.0 -framework Foundation -framework Metal -framework QuartzCore
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/mumax/3/cuda/metal"
)

// CreatePlan compiles a single-direction, in-place 2D R2C/C2R plan.
func CreatePlan(nx, ny, activeX, activeY int, inverse bool) (uintptr, error) {
	if err := metal.Initialize(); err != nil {
		return 0, fmt.Errorf("vkfft: initialize Metal runtime: %w", err)
	}
	var message *C.char
	handle := C.mvk_plan_create(
		C.int64_t(nx),
		C.int64_t(ny),
		C.int64_t(activeX),
		C.int64_t(activeY),
		boolInt(inverse),
		&message,
	)
	if handle == nil {
		return 0, bridgeError("create plan", message)
	}
	if message != nil {
		C.mvk_free_error(message)
	}
	return uintptr(handle), nil
}

// Execute appends one transform to the Metal runtime's current command buffer.
func Execute(handle, buffer uintptr, inverse bool) error {
	if handle == 0 || buffer == 0 {
		return fmt.Errorf("vkfft: invalid nil plan or buffer")
	}
	var message *C.char
	status := C.mvk_plan_append(
		unsafe.Pointer(handle),
		unsafe.Pointer(buffer),
		boolInt(inverse),
		&message,
	)
	if status != 0 {
		return bridgeError("execute", message)
	}
	if message != nil {
		C.mvk_free_error(message)
	}
	return nil
}

// DestroyPlan releases VkFFT pipelines and plan storage. The caller must drain
// the runtime first because command buffers retain those pipelines asynchronously.
func DestroyPlan(handle uintptr) {
	if handle != 0 {
		C.mvk_plan_destroy(unsafe.Pointer(handle))
	}
}

func boolInt(value bool) C.int {
	if value {
		return 1
	}
	return 0
}

func bridgeError(operation string, message *C.char) error {
	text := "unknown failure"
	if message != nil {
		text = C.GoString(message)
		C.mvk_free_error(message)
	}
	return fmt.Errorf("vkfft: %s: %s", operation, text)
}
