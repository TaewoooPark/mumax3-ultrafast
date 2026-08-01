package cuda

import (
	"math"
	"sync"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/timer"
	"github.com/mumax/3/util"
)

// Make a GPU Slice with nComp components each of size length.
func NewSlice(nComp int, size [3]int) *data.Slice {
	return newSlice(nComp, size, MemAlloc, data.GPUMemory)
}

// Make a GPU Slice with nComp components each of size length.
//func NewUnifiedSlice(nComp int, m *data.Mesh) *data.Slice {
//	return newSlice(nComp, m, cu.MemAllocHost, data.UnifiedMemory)
//}

// newSlice allocates all components as a single block so that elementwise
// kernels can cover them in one launch instead of one per component. See
// data.SliceFromContiguousPtrs.
func newSlice(nComp int, size [3]int, alloc func(int64) unsafe.Pointer, memType int8) *data.Slice {
	enableGPUSlices()
	length := prod(size)
	base := alloc(int64(length) * cu.SIZEOF_FLOAT32 * int64(nComp))
	cu.MemsetD32(cu.DevicePtr(uintptr(base)), 0, int64(length)*int64(nComp))
	return data.SliceFromContiguousPtrs(size, memType, componentPtrs(base, length, nComp))
}

// enableGPUSlices installs the backend callbacks needed by data.Slice.Free and
// data.Copy. Buffer and NewSlice are both public allocation entry points, so
// neither may rely on the other having been called first.
func enableGPUSlices() {
	enableGPUSlicesOnce.Do(func() {
		data.EnableGPU(memFree, cu.MemFreeHost, MemCpy, MemCpyDtoH, MemCpyHtoD)
	})
}

var enableGPUSlicesOnce sync.Once

// componentPtrs splits a block of nComp*length floats into component pointers.
func componentPtrs(base unsafe.Pointer, length, nComp int) []unsafe.Pointer {
	stride := uintptr(length) * cu.SIZEOF_FLOAT32
	ptrs := make([]unsafe.Pointer, nComp)
	for c := range ptrs {
		ptrs[c] = unsafe.Pointer(uintptr(base) + uintptr(c)*stride)
	}
	return ptrs
}

// wrappers for data.EnableGPU arguments

func memFree(ptr unsafe.Pointer) { cu.MemFree(cu.DevicePtr(uintptr(ptr))) }

func MemCpyDtoH(dst, src unsafe.Pointer, bytes int64) {
	// PROTOTYPE: the Metal runtime's copy-to-host already drains the queue
	// before reading the shared allocation, so the bracketing Sync() calls are
	// redundant full-pipeline stalls.
	timer.Start("memcpyDtoH")
	cu.MemcpyDtoH(dst, cu.DevicePtr(uintptr(src)), bytes)
	timer.Stop("memcpyDtoH")
}

func MemCpyHtoD(dst, src unsafe.Pointer, bytes int64) {
	// PROTOTYPE: mr_copy_to_device drains internally before the host write.
	timer.Start("memcpyHtoD")
	cu.MemcpyHtoD(cu.DevicePtr(uintptr(dst)), src, bytes)
	timer.Stop("memcpyHtoD")
}

func MemCpy(dst, src unsafe.Pointer, bytes int64) {
	// PROTOTYPE: device-to-device copy is a blit encoded into the same serial
	// queue as the surrounding kernels, so it is already ordered. Draining the
	// pipeline twice per copy costs ~2 GPU wake-ups for no ordering benefit.
	if Synchronous {
		Sync()
		timer.Start("memcpy")
	}
	cu.MemcpyAsync(cu.DevicePtr(uintptr(dst)), cu.DevicePtr(uintptr(src)), bytes, stream0)
	if Synchronous {
		Sync()
		timer.Stop("memcpy")
	}
}

// Memset sets the Slice's components to the specified values.
// To be carefully used on unified slice (need sync)
func Memset(s *data.Slice, val ...float32) {
	if Synchronous { // debug
		Sync()
		timer.Start("memset")
	}
	util.Argument(len(val) == s.NComp())
	if s.Contiguous() && equalFloats(val) {
		cu.MemsetD32Async(cu.DevicePtr(uintptr(s.DevPtr(0))), math.Float32bits(val[0]),
			int64(s.Len()*s.NComp()), stream0)
		if Synchronous {
			Sync()
			timer.Stop("memset")
		}
		return
	}
	for c, v := range val {
		cu.MemsetD32Async(cu.DevicePtr(uintptr(s.DevPtr(c))), math.Float32bits(v), int64(s.Len()), stream0)
	}
	if Synchronous { //debug
		Sync()
		timer.Stop("memset")
	}
}

func equalFloats(values []float32) bool {
	for _, value := range values[1:] {
		if value != values[0] {
			return false
		}
	}
	return true
}

// Set all elements of all components to zero.
func Zero(s *data.Slice) {
	if Synchronous {
		Sync()
		timer.Start("memset")
	}
	if s.Contiguous() {
		cu.MemsetD32Async(cu.DevicePtr(uintptr(s.DevPtr(0))), 0,
			int64(s.Len()*s.NComp()), stream0)
	} else {
		for c := 0; c < s.NComp(); c++ {
			cu.MemsetD32Async(cu.DevicePtr(uintptr(s.DevPtr(c))), 0,
				int64(s.Len()), stream0)
		}
	}
	if Synchronous {
		Sync()
		timer.Stop("memset")
	}
}

func SetCell(s *data.Slice, comp int, ix, iy, iz int, value float32) {
	SetElem(s, comp, s.Index(ix, iy, iz), value)
}

func SetElem(s *data.Slice, comp int, index int, value float32) {
	f := value
	dst := unsafe.Pointer(uintptr(s.DevPtr(comp)) + uintptr(index)*cu.SIZEOF_FLOAT32)
	MemCpyHtoD(dst, unsafe.Pointer(&f), cu.SIZEOF_FLOAT32)
}

func GetElem(s *data.Slice, comp int, index int) float32 {
	var f float32
	src := unsafe.Pointer(uintptr(s.DevPtr(comp)) + uintptr(index)*cu.SIZEOF_FLOAT32)
	MemCpyDtoH(unsafe.Pointer(&f), src, cu.SIZEOF_FLOAT32)
	return f
}

func GetCell(s *data.Slice, comp, ix, iy, iz int) float32 {
	return GetElem(s, comp, s.Index(ix, iy, iz))
}
