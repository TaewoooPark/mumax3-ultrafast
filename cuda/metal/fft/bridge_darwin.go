//go:build darwin && arm64 && cgo
// +build darwin,arm64,cgo

package fft

/*
#cgo darwin,arm64 CFLAGS: -mmacosx-version-min=14.0
#cgo darwin,arm64 CXXFLAGS: -x objective-c++ -std=c++17 -fobjc-arc -fblocks -mmacosx-version-min=14.0
#cgo darwin,arm64 LDFLAGS: -mmacosx-version-min=14.0 -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/mumax/3/cuda/metal"
	"github.com/mumax/3/cuda/metal/vkfft"
)

// Transform uses cuFFT's public numeric values so the compatibility package
// can pass its Type through without a translation table.
type Transform int32

const (
	ComplexToComplex Transform = C.MF_C2C
	RealToComplex    Transform = C.MF_R2C
	ComplexToReal    Transform = C.MF_C2R
)

type planRecord struct {
	mps     uintptr
	vk      uintptr
	inPlace bool
	inverse bool
}

var planRegistry = struct {
	sync.RWMutex
	next uintptr
	plan map[uintptr]planRecord
}{next: 1, plan: make(map[uintptr]planRecord)}

func registerPlan(plan planRecord) uintptr {
	planRegistry.Lock()
	defer planRegistry.Unlock()
	handle := planRegistry.next
	planRegistry.next++
	if planRegistry.next == 0 {
		planRegistry.next = 1
	}
	planRegistry.plan[handle] = plan
	return handle
}

// CreatePlan creates a cached FFT plan. Eligible 2D demagnetization plans use
// the separately compiled, non-ARC VkFFT bridge; every other shape keeps the
// established MPSGraph path.
func CreatePlan(layout Layout, transform Transform) (uintptr, error) {
	if err := metal.Initialize(); err != nil {
		return 0, fmt.Errorf("metal fft: initialize runtime: %w", err)
	}
	if vkFFTEligible(layout, transform) {
		inverse := transform == ComplexToReal
		handle, err := vkfft.CreatePlan(
			layout.Dimensions[2],
			layout.Dimensions[1],
			layout.ActiveInner,
			layout.ActiveOuter,
			inverse,
		)
		if err == nil {
			return registerPlan(planRecord{vk: handle, inPlace: true, inverse: inverse}), nil
		}
		if vkFFTForced() {
			return 0, fmt.Errorf("metal fft: forced VkFFT plan: %w", err)
		}
		// Auto mode is fail-closed: initialization errors retain the proven
		// MPSGraph implementation for this plan.
	}

	dimensions := make([]C.int64_t, len(layout.Dimensions))
	for i, dimension := range layout.Dimensions {
		dimensions[i] = C.int64_t(dimension)
	}
	var message *C.char
	handle := C.mf_plan_create(
		&dimensions[0],
		C.size_t(len(dimensions)),
		C.int64_t(layout.Batch),
		C.int32_t(transform),
		C.int64_t(layout.ActiveInner),
		C.int64_t(layout.ActiveOuter),
		&message,
	)
	runtime.KeepAlive(dimensions)
	if handle == nil {
		return 0, bridgeError("create plan", message)
	}
	if message != nil {
		C.mf_free_error(message)
	}
	return registerPlan(planRecord{mps: uintptr(handle)}), nil
}

// PlanIsInPlace reports whether a plan selected VkFFT's in-place R2C layout.
func PlanIsInPlace(handle uintptr) bool {
	planRegistry.RLock()
	defer planRegistry.RUnlock()
	return planRegistry.plan[handle].inPlace
}

// Execute appends a transform to the Metal runtime's current command buffer.
// MPSGraph may internally commit a large graph and continue on a replacement;
// the bridge returns that live root to the runtime so subsequent work remains
// ordered and Flush/Sync retain ownership of the final batch.
func Execute(handle, input, output uintptr, direction int) error {
	if handle == 0 || input == 0 || output == 0 {
		return fmt.Errorf("metal fft: invalid nil plan or buffer")
	}
	planRegistry.RLock()
	defer planRegistry.RUnlock()
	plan, ok := planRegistry.plan[handle]
	if !ok {
		return fmt.Errorf("metal fft: unknown or destroyed plan %d", handle)
	}
	if plan.inPlace {
		if input != output {
			return fmt.Errorf("metal fft: VkFFT R2C/C2R plan requires one in-place buffer")
		}
		return vkfft.Execute(plan.vk, input, plan.inverse)
	}
	var message *C.char
	status := C.mf_plan_execute(
		unsafe.Pointer(plan.mps),
		unsafe.Pointer(input),
		unsafe.Pointer(output),
		C.int32_t(direction),
		&message,
	)
	if status != C.MF_SUCCESS {
		return bridgeError("execute", message)
	}
	if message != nil {
		C.mf_free_error(message)
	}
	return nil
}

func forceCommitAndContinueForTest() error {
	var message *C.char
	status := C.mf_test_commit_and_continue(&message)
	if status != C.MF_SUCCESS {
		return bridgeError("test commit-and-continue", message)
	}
	if message != nil {
		C.mf_free_error(message)
	}
	return nil
}

// DestroyPlan releases graph and descriptor objects cached by a plan.
func DestroyPlan(handle uintptr) error {
	if handle == 0 {
		return nil
	}
	planRegistry.Lock()
	defer planRegistry.Unlock()
	plan, ok := planRegistry.plan[handle]
	if !ok {
		return nil
	}
	delete(planRegistry.plan, handle)

	// MPSGraph command buffers may retain references into the graph until GPU
	// completion, and VkFFT pipelines have the same asynchronous lifetime. Plan
	// destruction is rare, so synchronize before releasing either backend.
	syncErr := metal.Sync()
	if plan.inPlace {
		vkfft.DestroyPlan(plan.vk)
		if syncErr != nil {
			return fmt.Errorf("metal fft: synchronize before destroying plan: %w", syncErr)
		}
		return nil
	}
	var message *C.char
	status := C.mf_plan_destroy(unsafe.Pointer(plan.mps), &message)
	if status != C.MF_SUCCESS {
		return bridgeError("destroy plan", message)
	}
	if message != nil {
		C.mf_free_error(message)
	}
	if syncErr != nil {
		return fmt.Errorf("metal fft: synchronize before destroying plan: %w", syncErr)
	}
	return nil
}

func vkFFTEligible(layout Layout, transform Transform) bool {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("MUMAX3_METAL_FFT_BACKEND")))
	if mode == "mps" || mode == "mpsgraph" {
		return false
	}
	if len(layout.Dimensions) != 3 || layout.Batch != 1 ||
		layout.Dimensions[0] != 1 ||
		(transform != RealToComplex && transform != ComplexToReal) {
		return false
	}
	ny, nx := layout.Dimensions[1], layout.Dimensions[2]
	// The end-to-end crossover is size-dependent: padded extents through 512
	// showed a material win, while 1024 was only about 1.6% faster and carried
	// a larger first-plan compilation cost. Keep that marginal tier opt-in.
	maximum := 512
	if mode == "vkfft" {
		maximum = 1024
	}
	return nx > 1 && nx <= maximum && ny > 1 && ny <= maximum &&
		isPowerOfTwo(nx) && isPowerOfTwo(ny) &&
		layout.ActiveInner > 0 && layout.ActiveInner < nx &&
		layout.ActiveOuter > 0 && layout.ActiveOuter < ny
}

func isPowerOfTwo(value int) bool {
	return value > 0 && value&(value-1) == 0
}

func vkFFTForced() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv("MUMAX3_METAL_FFT_BACKEND")), "vkfft")
}

func bridgeError(operation string, message *C.char) error {
	text := "unknown failure"
	if message != nil {
		text = C.GoString(message)
		C.mf_free_error(message)
	}
	return fmt.Errorf("metal fft: %s: %s", operation, text)
}
