//go:build darwin && arm64 && cgo

package cuda

import (
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// BBStepSupported reports whether the Barzilai-Borwein step size can stay on the
// device. It needs the bbminimize kernel, whose CUDA launch wrapper is generated
// by cuda/Makefile and therefore requires nvcc; only the Metal wrapper is checked
// in, so the CUDA build keeps computing the step size on the host.
const BBStepSupported = true

// BBStep keeps the un-combined partial slots of the two dot products the
// Barzilai-Borwein step size is built from, so that the scalar never has to
// travel through the host.
//
// Reading it back is what makes the steepest-descent minimizer spend one full
// pipeline drain per iteration: the next descent cannot be encoded until the
// previous iteration's reductions have landed. Leaving the partials on the
// device removes the dependency entirely - queue order alone guarantees the
// reductions of iteration n complete before the descent of iteration n+1 reads
// them.
type BBStep struct {
	nom   unsafe.Pointer
	div   unsafe.Pointer
	slots int
}

// NewBBStep allocates the partial-slot buffers and seeds them with the same
// initial step size the host path starts from.
func NewBBStep() *BBStep {
	slots := 3 * reduceSumSlots
	step := &BBStep{
		nom:   MemAlloc(int64(slots) * cu.SIZEOF_FLOAT32),
		div:   MemAlloc(int64(slots) * cu.SIZEOF_FLOAT32),
		slots: slots,
	}
	step.Seed(1e-4)
	return step
}

// Seed makes the next descent use exactly h, by putting h in the first
// numerator slot and 1 in the first denominator slot and zeroing the rest. The
// kernel adds the slots in index order, so the quotient is exactly h.
func (step *BBStep) Seed(h float32) {
	zeroFloats(step.nom, step.slots)
	zeroFloats(step.div, step.slots)
	setFloat(step.nom, h)
	setFloat(step.div, 1)
}

// Update enqueues the two dot products for the next step size. It alternates
// between the two Barzilai-Borwein formulas exactly as the host path does.
func (step *BBStep) Update(dm, dk *data.Slice, longFormula bool) {
	if longFormula {
		step.dotInto(step.nom, dm, dm)
		step.dotInto(step.div, dm, dk)
	} else {
		step.dotInto(step.nom, dm, dk)
		step.dotInto(step.div, dk, dk)
	}
}

// dotInto enqueues the same per-component reductions DotAsync issues, into a
// caller-owned buffer that is never copied back. Each component keeps its own
// block of slots, so the sum stays independent of threadgroup scheduling and of
// overlap between launches, exactly as in DotAsync.
func (step *BBStep) dotInto(dst unsafe.Pointer, a, b *data.Slice) {
	nComp := a.NComp()
	util.Argument(nComp == b.NComp())
	util.Argument(nComp*reduceSumSlots == step.slots)
	cu.MemsetD32Async(cu.DevicePtr(uintptr(dst)), 0,
		int64(step.slots), stream0)
	for c := 0; c < nComp; c++ {
		k_reducedot_async(a.DevPtr(c), b.DevPtr(c),
			slot(dst, c, reduceSumSlots), 0, a.Len(), reducesumcfg)
	}
}

func (step *BBStep) Free() {
	if step == nil {
		return
	}
	if step.nom != nil {
		MemFree(step.nom)
		step.nom = nil
	}
	if step.div != nil {
		MemFree(step.div)
		step.div = nil
	}
}

// MinimizeBB is Minimize with the step size read from the device.
func MinimizeBB(m, m0, torque *data.Slice, step *BBStep) {
	N := m.Len()
	cfg := make1DConf(N)

	k_bbminimize_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		step.nom, step.div, step.slots, N, cfg)
}

func zeroFloats(dst unsafe.Pointer, count int) {
	cu.MemsetD32Async(cu.DevicePtr(uintptr(dst)), 0, int64(count), stream0)
}

// setFloat writes one float to the head of a device allocation. Shared memory
// makes this a plain store, but it still has to be ordered against work already
// encoded, so it goes through the ordered upload.
func setFloat(dst unsafe.Pointer, value float32) {
	MemCpyHtoD(dst, unsafe.Pointer(&value), cu.SIZEOF_FLOAT32)
}
