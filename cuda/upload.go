package cuda

import (
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
)

// MemCpyHtoDUnordered uploads without first waiting for encoded work to
// finish.
//
// The Metal backend keeps its allocations in shared memory, so a plain upload
// has to drain the whole queue before writing: the store would otherwise be
// seen by kernels that were encoded earlier and are still running. Draining is
// far more expensive than the upload itself, and it is only necessary because
// the destination may be in use. Callers that can rule that out - by writing a
// slot they have just rotated to, for instance - should use this instead.
//
// On CUDA there is nothing to relax; the driver already orders uploads against
// the default stream.
func MemCpyHtoDUnordered(dst, src unsafe.Pointer, bytes int64) {
	timer.Start("memcpyHtoD")
	cu.MemcpyHtoDUnordered(cu.DevicePtr(uintptr(dst)), src, bytes)
	timer.Stop("memcpyHtoD")
}
