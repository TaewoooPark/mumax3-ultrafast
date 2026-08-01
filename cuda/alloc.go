package cuda

import (
	"log"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
)

// Wrapper for cu.MemAlloc, fatal exit on out of memory.
func MemAlloc(bytes int64) unsafe.Pointer {
	defer func() {
		err := recover()
		if err == cu.ERROR_OUT_OF_MEMORY {
			log.Fatal(err)
		}
		if err != nil {
			panic(err)
		}
	}()
	return unsafe.Pointer(uintptr(cu.MemAlloc(bytes)))
}

// MemFree releases a base pointer returned by MemAlloc. Interior pointers are
// not valid. Most simulation storage is owned by data.Slice; this entry point
// is for small backend-neutral allocations such as region LUT rings.
func MemFree(pointer unsafe.Pointer) {
	if pointer != nil {
		cu.MemFree(cu.DevicePtr(uintptr(pointer)))
	}
}

// Returns a copy of in, allocated on GPU.
func GPUCopy(in *data.Slice) *data.Slice {
	s := NewSlice(in.NComp(), in.Size())
	data.Copy(s, in)
	return s
}
