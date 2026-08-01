//go:build darwin && arm64 && cgo

package metal

/*
#cgo darwin,arm64 CFLAGS: -mmacosx-version-min=14.0
#cgo darwin,arm64 CXXFLAGS: -x objective-c++ -std=c++17 -fobjc-arc -fblocks -mmacosx-version-min=14.0
#cgo darwin,arm64 LDFLAGS: -mmacosx-version-min=14.0 -framework Foundation -framework Metal
#include <stdlib.h>
#include "metal_runtime.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

func Available() bool {
	return Initialize() == nil
}

func Initialize() error {
	return runtimeStatus(C.mr_initialize(nil), nil)
}

func Close() error {
	var message *C.char
	return runtimeStatus(C.mr_shutdown(&message), message)
}

func Info() (DeviceInfo, error) {
	var info C.mr_device_info
	var message *C.char
	status := C.mr_get_device_info(&info, &message)
	if err := runtimeStatus(status, message); err != nil {
		return DeviceInfo{}, err
	}
	return DeviceInfo{
		Name:                         C.GoString(&info.name[0]),
		UnifiedMemory:                info.unified_memory != 0,
		MaxThreadsPerThreadgroup:     [3]uint64{uint64(info.max_threads_x), uint64(info.max_threads_y), uint64(info.max_threads_z)},
		RecommendedMaxWorkingSetSize: uint64(info.recommended_max_working_set_size),
		CurrentAllocatedSize:         uint64(info.current_allocated_size),
		TrackedAllocationSize:        uint64(info.tracked_allocation_size),
		TrackedPeakAllocationSize:    uint64(info.tracked_peak_allocation_size),
	}, nil
}

func RegisterSource(source string) error {
	if source == "" {
		return errors.New("mumax3/metal: refusing to register empty Metal source")
	}
	data := C.CBytes([]byte(source))
	defer C.free(data)
	var message *C.char
	return runtimeStatus(
		C.mr_register_source(data, C.size_t(len(source)), &message),
		message,
	)
}

func RegisterLibrary(library []byte) error {
	if len(library) == 0 {
		return errors.New("mumax3/metal: refusing to register an empty metallib")
	}
	data := C.CBytes(library)
	defer C.free(data)
	var message *C.char
	return runtimeStatus(
		C.mr_register_library(data, C.size_t(len(library)), &message),
		message,
	)
}

func Alloc(bytes int64) (unsafe.Pointer, error) {
	if bytes <= 0 {
		return nil, errors.New("mumax3/metal: allocation size must be positive")
	}
	if uint64(bytes) > uint64(^C.size_t(0)) {
		return nil, errors.New("mumax3/metal: allocation size exceeds platform size_t")
	}
	var pointer unsafe.Pointer
	var message *C.char
	status := C.mr_alloc(C.size_t(bytes), &pointer, &message)
	return pointer, runtimeStatus(status, message)
}

func MustAlloc(bytes int64) unsafe.Pointer {
	pointer, err := Alloc(bytes)
	if err != nil {
		panic(err)
	}
	return pointer
}

func Free(pointer unsafe.Pointer) error {
	if pointer == nil {
		return nil
	}
	var message *C.char
	return runtimeStatus(C.mr_free(pointer, &message), message)
}

func AllocationRange(pointer unsafe.Pointer) (unsafe.Pointer, int64, error) {
	if pointer == nil {
		return nil, 0, errors.New("mumax3/metal: allocation query pointer is nil")
	}
	var base unsafe.Pointer
	var bytes C.size_t
	var message *C.char
	status := C.mr_get_address_range(pointer, &base, &bytes, &message)
	if err := runtimeStatus(status, message); err != nil {
		return nil, 0, err
	}
	return base, int64(bytes), nil
}

func Copy(dst, src unsafe.Pointer, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if err := validCopy(dst, src, bytes); err != nil {
		return err
	}
	var message *C.char
	return runtimeStatus(C.mr_copy(dst, src, C.size_t(bytes), &message), message)
}

func CopyToDevice(dst, src unsafe.Pointer, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if err := validCopy(dst, src, bytes); err != nil {
		return err
	}
	var message *C.char
	return runtimeStatus(C.mr_copy_to_device(dst, src, C.size_t(bytes), &message), message)
}

// CopyToDeviceUnordered writes dst without draining the queue first. The
// caller must guarantee that no encoded work reads the destination range; see
// mr_copy_to_device_unordered.
func CopyToDeviceUnordered(dst, src unsafe.Pointer, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if err := validCopy(dst, src, bytes); err != nil {
		return err
	}
	var message *C.char
	return runtimeStatus(C.mr_copy_to_device_unordered(dst, src, C.size_t(bytes), &message), message)
}

func CopyToHost(dst, src unsafe.Pointer, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if err := validCopy(dst, src, bytes); err != nil {
		return err
	}
	var message *C.char
	return runtimeStatus(C.mr_copy_to_host(dst, src, C.size_t(bytes), &message), message)
}

// CopyToHostUnordered reads src without draining the queue first. The caller
// must have proven separately that every kernel writing the source range has
// finished, typically by waiting a Completion recorded right after they were
// encoded. See mr_copy_to_host_unordered.
func CopyToHostUnordered(dst, src unsafe.Pointer, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if err := validCopy(dst, src, bytes); err != nil {
		return err
	}
	var message *C.char
	return runtimeStatus(C.mr_copy_to_host_unordered(dst, src, C.size_t(bytes), &message), message)
}

func Fill(dst unsafe.Pointer, value byte, bytes int64) error {
	if bytes == 0 {
		return nil
	}
	if dst == nil || bytes < 0 {
		return errors.New("mumax3/metal: invalid fill range")
	}
	var message *C.char
	return runtimeStatus(C.mr_fill(dst, C.uint8_t(value), C.size_t(bytes), &message), message)
}

func FillUint32(dst unsafe.Pointer, value uint32, count int64) error {
	if count == 0 {
		return nil
	}
	if dst == nil || count < 0 {
		return errors.New("mumax3/metal: invalid uint32 fill range")
	}
	var message *C.char
	return runtimeStatus(
		C.mr_fill_u32(dst, C.uint32_t(value), C.size_t(count), &message),
		message,
	)
}

// maxKernelArguments is the Metal buffer-table limit for a compute kernel.
const maxKernelArguments = 31

func encodeGrid(cfg GridConfig) C.mr_grid {
	return C.mr_grid{
		grid_x:    C.uint32_t(cfg.GridX),
		grid_y:    C.uint32_t(cfg.GridY),
		grid_z:    C.uint32_t(cfg.GridZ),
		block_x:   C.uint32_t(cfg.BlockX),
		block_y:   C.uint32_t(cfg.BlockY),
		block_z:   C.uint32_t(cfg.BlockZ),
		threads_x: C.uint32_t(cfg.ThreadsX),
		threads_y: C.uint32_t(cfg.ThreadsY),
		threads_z: C.uint32_t(cfg.ThreadsZ),
	}
}

// encodeArgs fills a caller-owned array so the dispatch path does not allocate.
// The array is a value type holding only scalars and pointers into Metal
// allocations, so it never needs to escape to the heap.
func encodeArgs(dst *[maxKernelArguments]C.mr_arg, args []Arg) *C.mr_arg {
	if len(args) == 0 {
		return nil
	}
	for i, arg := range args {
		dst[i].kind = C.uint32_t(arg.kind)
		dst[i].size = C.uint32_t(arg.size)
		dst[i].buffer = arg.pointer
		dst[i].bits = C.uint64_t(arg.bits)
	}
	return &dst[0]
}

func Launch(name string, cfg GridConfig, args ...Arg) error {
	if name == "" {
		return errors.New("mumax3/metal: kernel name is empty")
	}
	if cfg.BlockX == 0 || cfg.BlockY == 0 || cfg.BlockZ == 0 {
		return errors.New("mumax3/metal: threadgroup dimensions must be non-zero")
	}
	if len(args) > maxKernelArguments {
		return errors.New("mumax3/metal: a Metal compute kernel cannot bind more than 31 buffer-table arguments")
	}

	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	var storage [maxKernelArguments]C.mr_arg
	var message *C.char
	return runtimeStatus(
		C.mr_launch(cname, encodeGrid(cfg), encodeArgs(&storage, args),
			C.size_t(len(args)), &message),
		message,
	)
}

func MustLaunch(name string, cfg GridConfig, args ...Arg) {
	if err := Launch(name, cfg, args...); err != nil {
		panic(err)
	}
}

// Kernel caches the runtime handle of one compute kernel. A single RK45 step
// issues well over a hundred dispatches, so resolving the name once removes a
// C string allocation, an NSString allocation and a dictionary hash from every
// one of them.
type Kernel struct {
	name     string
	handle   uint32
	resolved bool
}

// NewKernel names a kernel without resolving it. Resolution is deferred to the
// first launch so that package-level kernel variables can be declared before
// the Metal device and shader library exist.
func NewKernel(name string) *Kernel { return &Kernel{name: name} }

// Name reports the kernel name this handle was created for.
func (k *Kernel) Name() string { return k.name }

func (k *Kernel) resolve() error {
	if k.resolved {
		return nil
	}
	if k.name == "" {
		return errors.New("mumax3/metal: kernel name is empty")
	}
	cname := C.CString(k.name)
	defer C.free(unsafe.Pointer(cname))
	var handle C.uint32_t
	var message *C.char
	if err := runtimeStatus(
		C.mr_register_kernel(cname, &handle, &message), message,
	); err != nil {
		return err
	}
	k.handle = uint32(handle)
	k.resolved = true
	return nil
}

// Launch dispatches this kernel through its cached handle.
func (k *Kernel) Launch(cfg GridConfig, args ...Arg) error {
	if cfg.BlockX == 0 || cfg.BlockY == 0 || cfg.BlockZ == 0 {
		return errors.New("mumax3/metal: threadgroup dimensions must be non-zero")
	}
	if len(args) > maxKernelArguments {
		return errors.New("mumax3/metal: a Metal compute kernel cannot bind more than 31 buffer-table arguments")
	}
	if err := k.resolve(); err != nil {
		return err
	}
	var storage [maxKernelArguments]C.mr_arg
	var message *C.char
	return runtimeStatus(
		C.mr_launch_handle(C.uint32_t(k.handle), encodeGrid(cfg),
			encodeArgs(&storage, args), C.size_t(len(args)), &message),
		message,
	)
}

// MustLaunch is Launch with a panic on failure, matching MuMax3's convention
// that a kernel dispatch failure is not recoverable.
func (k *Kernel) MustLaunch(cfg GridConfig, args ...Arg) {
	if err := k.Launch(cfg, args...); err != nil {
		panic(err)
	}
}

func Flush() error {
	var message *C.char
	return runtimeStatus(C.mr_flush(&message), message)
}

func Sync() error {
	var message *C.char
	return runtimeStatus(C.mr_synchronize(&message), message)
}

// Completion retains the exact Metal command buffer that was the ordered
// queue tail when RecordCompletion was called. It does not submit work merely
// by being recorded.
type Completion struct {
	handle unsafe.Pointer
}

func RecordCompletion() (Completion, error) {
	var handle unsafe.Pointer
	var message *C.char
	status := C.mr_record_completion(&handle, &message)
	if err := runtimeStatus(status, message); err != nil {
		return Completion{}, err
	}
	return Completion{handle: handle}, nil
}

func (completion Completion) Valid() bool { return completion.handle != nil }

// Ready reports completion without submitting or waiting for any work.
func (completion Completion) Ready() (bool, error) {
	if completion.handle == nil {
		return true, nil
	}
	var ready C.int
	var message *C.char
	status := C.mr_query_completion(completion.handle, &ready, &message)
	return ready != 0, runtimeStatus(status, message)
}

// Wait waits only the retained command buffer. If that buffer is still the
// runtime's live root, Wait submits it first; an MPS-adopted stale root is
// already committed and is never committed again.
func (completion Completion) Wait() error {
	if completion.handle == nil {
		return nil
	}
	var message *C.char
	return runtimeStatus(
		C.mr_wait_completion(completion.handle, &message),
		message,
	)
}

func (completion *Completion) Close() {
	if completion == nil || completion.handle == nil {
		return
	}
	C.mr_release_completion(completion.handle)
	completion.handle = nil
}

func GetRuntimeStats() (RuntimeStats, error) {
	var stats C.mr_runtime_stats
	var message *C.char
	status := C.mr_get_runtime_stats(&stats, &message)
	if err := runtimeStatus(status, message); err != nil {
		return RuntimeStats{}, err
	}
	return RuntimeStats{
		CommandBufferSubmissions:  uint64(stats.command_buffer_submissions),
		ExternalRootAdoptions:     uint64(stats.external_root_adoptions),
		FullDrains:                uint64(stats.full_drains),
		CompletionRecords:         uint64(stats.completion_records),
		CompletionQueries:         uint64(stats.completion_queries),
		CompletionQueryHits:       uint64(stats.completion_query_hits),
		CompletionWaits:           uint64(stats.completion_waits),
		CompletionWaitSubmissions: uint64(stats.completion_wait_submissions),
		KeepAliveSubmissions:      uint64(stats.keepalive_submissions),
	}, nil
}

func ResetRuntimeStats() error {
	var message *C.char
	return runtimeStatus(C.mr_reset_runtime_stats(&message), message)
}

func validCopy(dst, src unsafe.Pointer, bytes int64) error {
	if dst == nil || src == nil || bytes < 0 {
		return errors.New("mumax3/metal: invalid copy range")
	}
	return nil
}

func runtimeStatus(status C.int, message *C.char) error {
	if message != nil {
		defer C.mr_free_error(message)
	}
	if status == C.MR_SUCCESS {
		return nil
	}
	text := "unknown Metal runtime failure"
	if message != nil {
		text = C.GoString(message)
	}
	return &RuntimeError{Code: int(status), Message: text}
}
