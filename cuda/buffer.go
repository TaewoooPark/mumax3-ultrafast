package cuda

// Pool of re-usable GPU buffers.
// Synchronization subtlety:
// async kernel launches mean a buffer may already be recycled when still in use.
// That should be fine since the next launch runs in the same stream (0), and will
// effectively wait for the previous operation on the buffer.

import (
	"log"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
)

// A pooled buffer holds all of its components in one allocation, so that
// elementwise kernels can cover them in a single launch. The pool is therefore
// keyed by shape rather than by component: a block is handed out and taken back
// whole, which is what keeps the components adjacent for the lifetime of the
// process.
type bufShape struct {
	N     int
	nComp int
}

var (
	buf_pool  = make(map[bufShape][]unsafe.Pointer) // pool of GPU blocks indexed by shape
	buf_check = make(map[unsafe.Pointer]struct{})   // checks if pointer originates here to avoid unintended recycle
)

const buf_max = 100 // maximum number of buffers to allocate (detect memory leak early)

// Returns a GPU slice for temporary use. To be returned to the pool with Recycle
func Buffer(nComp int, size [3]int) *data.Slice {
	if Synchronous {
		Sync()
	}

	N := prod(size)
	shape := bufShape{N, nComp}
	pool := buf_pool[shape]

	var base unsafe.Pointer
	if len(pool) > 0 {
		base = pool[len(pool)-1]
		buf_pool[shape] = pool[:len(pool)-1]
	} else {
		// Counts distinct addresses, not allocations: a caller that frees a
		// pooled block outside Recycle (Minimizer.Free does) gets the same
		// address back from the allocator, and that must not read as a leak.
		if len(buf_check) >= buf_max {
			log.Panic("too many buffers in use, possible memory leak")
		}
		base = MemAlloc(int64(cu.SIZEOF_FLOAT32 * N * nComp))
		buf_check[base] = struct{}{} // mark this pointer as mine
	}

	return data.SliceFromContiguousPtrs(size, data.GPUMemory, componentPtrs(base, N, nComp))
}

// Returns a buffer obtained from GetBuffer to the pool.
func Recycle(s *data.Slice) {
	if Synchronous {
		Sync()
	}

	if s.NComp() == 0 {
		return
	}
	shape := bufShape{s.Len(), s.NComp()}
	base := s.DevPtr(0)
	if base != unsafe.Pointer(uintptr(0)) {
		if _, ok := buf_check[base]; !ok {
			log.Panic("recyle: was not obtained with getbuffer")
		}
		buf_pool[shape] = append(buf_pool[shape], base)
	}
	s.Disable() // make it unusable, protect against accidental use after recycle
}

// Frees all buffers. Called after mesh resize.
func FreeBuffers() {
	Sync()
	for _, blocks := range buf_pool {
		for i := range blocks {
			cu.DevicePtr(uintptr(blocks[i])).Free()
			blocks[i] = nil
		}
	}
	buf_pool = make(map[bufShape][]unsafe.Pointer)
	buf_check = make(map[unsafe.Pointer]struct{})
}
