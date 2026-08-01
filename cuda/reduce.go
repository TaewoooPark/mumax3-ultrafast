package cuda

import (
	"math"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Block size for reduce kernels.
const REDUCE_BLOCKSIZE = 512

// Number of threadgroups in a reduction, and therefore the number of partial
// results it writes. Each threadgroup owns one slot; the host combines the slots
// in fixed index order so the result is reproducible.
//
// The two families use different widths on purpose. Maximum is order
// independent in floating point, so widening it past the eight threadgroups
// CUDA used cannot change the result and simply uses more of the GPU: on an M4,
// 64 threadgroups beat 8 by 1.3x to 1.6x, and 256 is slower again. Addition is
// not associative, so the sum reductions keep eight slots; widening them would
// change the last bits of the energy and move where relax() stops.
const (
	reduceMaxSlots = 64
	reduceSumSlots = 8
)

// Sum of all elements.
func Sum(in *data.Slice) float32 {
	return float32(SumAsync(in).Value())
}

// SumAsync enqueues Sum without reading its partials back. Several independent
// reductions can be launched first and then resolved with one queue drain.
func SumAsync(in *data.Slice) *Pending {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0, 1, reduceSumSlots)
	k_reducesum_async(in.DevPtr(0), out, 0, in.Len(), reducesumcfg)
	return &Pending{buf: out, nComp: 1, slots: reduceSumSlots, sum: true}
}

// SumComponentsAsync enqueues one independent sum per component. Values reads
// all component partials back in one transfer, preserving the same per-
// component addition order as separate Sum calls.
func SumComponentsAsync(in *data.Slice) *PendingComponents {
	nComp := in.NComp()
	out := reduceBuf(0, nComp, reduceSumSlots)
	for c := 0; c < nComp; c++ {
		k_reducesum_async(in.DevPtr(c), slot(out, c, reduceSumSlots), 0, in.Len(), reducesumcfg)
	}
	return &PendingComponents{buf: out, nComp: nComp, slots: reduceSumSlots}
}

// Dot product.
func Dot(a, b *data.Slice) float32 {
	return float32(DotAsync(a, b).Value())
}

// DotAsync is the deferred-readback counterpart of Dot.
func DotAsync(a, b *data.Slice) *Pending {
	nComp := a.NComp()
	util.Argument(nComp == b.NComp())
	out := reduceBuf(0, nComp, reduceSumSlots)
	// Not async over components. Each component writes its own block of slots
	// rather than accumulating into a shared one, so the sum stays independent
	// of both threadgroup scheduling and inter-launch overlap.
	for c := 0; c < nComp; c++ {
		k_reducedot_async(a.DevPtr(c), b.DevPtr(c), slot(out, c, reduceSumSlots), 0, a.Len(), reducesumcfg)
	}
	return &Pending{buf: out, nComp: nComp, slots: reduceSumSlots, sum: true}
}

// DotComponentsAsync enqueues one dot product per component. b may either
// match a's component count or be scalar, in which case that scalar component
// is paired with every component of a. Values performs a single readback.
func DotComponentsAsync(a, b *data.Slice) *PendingComponents {
	nComp := a.NComp()
	util.Argument(b.NComp() == nComp || b.NComp() == 1)
	util.Argument(a.Len() == b.Len())
	out := reduceBuf(0, nComp, reduceSumSlots)
	for c := 0; c < nComp; c++ {
		bc := c
		if b.NComp() == 1 {
			bc = 0
		}
		k_reducedot_async(a.DevPtr(c), b.DevPtr(bc), slot(out, c, reduceSumSlots), 0, a.Len(), reducesumcfg)
	}
	return &PendingComponents{buf: out, nComp: nComp, slots: reduceSumSlots}
}

// Maximum of absolute values of all elements.
func MaxAbs(in *data.Slice) float32 {
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0, 1, reduceMaxSlots)
	k_reducemaxabs_async(in.DevPtr(0), out, 0, in.Len(), reducemaxcfg)
	return copybackMax(out, 1, reduceMaxSlots)
}

// Maximum of the norms of all vectors (x[i], y[i], z[i]).
//
//	max_i sqrt( x[i]*x[i] + y[i]*y[i] + z[i]*z[i] )
func MaxVecNorm(v *data.Slice) float64 {
	return MaxVecNormAsync(v).Value()
}

// Pending is a reduction whose kernel has been enqueued but whose partial
// results have not been copied back yet.
//
// Reading a reduction is what forces the GPU pipeline to drain, and on a
// latency-bound backend that costs far more than the reduction itself. A
// caller that only needs the number for reporting can keep a Pending and
// resolve it later, when a drain happens anyway. The value is unchanged by
// waiting: the partial slots are owned by this Pending until Value reads them.
type Pending struct {
	buf    unsafe.Pointer
	nComp  int
	slots  int
	sqrt   bool
	sum    bool
	value  float64
	loaded bool
	// Set by Targeted. Marks this reduction's place in the queue so Value can
	// wait for just that work instead of everything encoded behind it.
	boundary Completion
}

// PendingComponents is a component-wise family of sum reductions sharing one
// allocation and one host readback.
type PendingComponents struct {
	buf    unsafe.Pointer
	nComp  int
	slots  int
	values []float64
}

// Values resolves every component in one transfer. It is idempotent.
func (p *PendingComponents) Values() []float64 {
	if p.values == nil {
		partial := copybackPartials(p.buf, p.nComp, p.slots)
		p.values = make([]float64, p.nComp)
		for c := 0; c < p.nComp; c++ {
			var sum float32
			for _, value := range partial[c*p.slots : (c+1)*p.slots] {
				sum += value
			}
			p.values[c] = float64(sum)
		}
	}
	return p.values
}

// MaxTracker keeps the most recent maximum reduction, and optionally the
// maximum observed across many launches, entirely on the device. It is meant
// for diagnostic values that are reported later but must not make the CPU
// drain the command queue every step.
//
// The first reduceMaxSlots floats hold the latest launch's partial maxima.
// The final float accumulates their maximum over time. Queue ordering makes it
// safe to overwrite/accumulate these locations while earlier kernels are still
// in flight.
type MaxTracker struct {
	buf  unsafe.Pointer
	peak unsafe.Pointer
}

// NewMaxTracker allocates a persistent tracker. Call Free when the tracker is
// not process-long-lived.
func NewMaxTracker() *MaxTracker {
	buf := MemAlloc((reduceMaxSlots + 1) * cu.SIZEOF_FLOAT32)
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), 0, reduceMaxSlots+1, stream0)
	return &MaxTracker{
		buf:  buf,
		peak: unsafe.Pointer(uintptr(buf) + uintptr(reduceMaxSlots)*cu.SIZEOF_FLOAT32),
	}
}

// TrackMaxVecNorm replaces the latest value with max_i |v[i]| and, when
// accumulatePeak is true, folds it into the tracker-wide peak.
func (t *MaxTracker) TrackMaxVecNorm(v *data.Slice, accumulatePeak bool) {
	util.Argument(v.NComp() == 3)
	t.clearLatest()
	k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2),
		t.buf, 0, v.Len(), reducemaxcfg)
	if accumulatePeak {
		t.accumulatePeak()
	}
}

// TrackMaxVecDiff is the difference-vector counterpart of TrackMaxVecNorm.
func (t *MaxTracker) TrackMaxVecDiff(x, y *data.Slice, accumulatePeak bool) {
	util.Argument(x.NComp() == 3 && y.NComp() == 3 && x.Len() == y.Len())
	t.clearLatest()
	k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		t.buf, 0, x.Len(), reducemaxcfg)
	if accumulatePeak {
		t.accumulatePeak()
	}
}

func (t *MaxTracker) clearLatest() {
	util.Assert(t != nil && t.buf != nil)
	cu.MemsetD32Async(cu.DevicePtr(uintptr(t.buf)), 0, reduceMaxSlots, stream0)
}

func (t *MaxTracker) accumulatePeak() {
	// One 64-thread block reduces the 64 latest partials. The destination is
	// deliberately not cleared: reducemaxabs uses atomic max, so peak[0]
	// becomes the exact maximum across every tracked launch.
	k_reducemaxabs_async(t.buf, t.peak, 0, reduceMaxSlots, reducePeakCfg)
}

// Values reads the latest maximum and accumulated peak. Both returned values
// are vector norms (the square root is applied after the max, just like
// MaxVecNorm). When resetPeak is true, subsequent tracking starts a new peak
// segment after this read.
func (t *MaxTracker) Values(resetPeak bool) (latest, peak float64) {
	util.Assert(t != nil && t.buf != nil)
	partial := make([]float32, reduceMaxSlots+1)
	MemCpyDtoH(unsafe.Pointer(&partial[0]), t.buf,
		int64(len(partial))*cu.SIZEOF_FLOAT32)
	latest2 := partial[0]
	for _, value := range partial[1:reduceMaxSlots] {
		if value > latest2 {
			latest2 = value
		}
	}
	latest = math.Sqrt(float64(latest2))
	peak = math.Sqrt(float64(partial[reduceMaxSlots]))
	if resetPeak {
		cu.MemsetD32Async(cu.DevicePtr(uintptr(t.peak)), 0, 1, stream0)
	}
	return latest, peak
}

func (t *MaxTracker) Free() {
	if t == nil || t.buf == nil {
		return
	}
	cu.MemFree(cu.DevicePtr(uintptr(t.buf)))
	t.buf = nil
	t.peak = nil
}

// Targeted closes the current queue batch and records where this reduction sits
// in it, so that Value waits only for the work up to this point. Everything
// encoded afterwards keeps running while the host waits, which is only useful if
// the caller actually has later work to encode - otherwise the wait is the same
// length and the extra batch boundary is pure cost. Returns p for chaining.
func (p *Pending) Targeted() *Pending {
	if !TargetedReadbackSupported {
		return p
	}
	p.boundary = RecordCompletion()
	CloseQueueBatch()
	return p
}

// Value copies the partial results back, combines them and recycles the
// reduction buffer. It is idempotent.
func (p *Pending) Value() float64 {
	if !p.loaded {
		var v float64
		if p.boundary.Valid() {
			// Wait only this reduction's own batch, then read the shared
			// allocation directly. The combining order over the fixed slots is
			// the same either way, so the value is unchanged.
			partial := make([]float32, p.nComp*p.slots)
			ReadHostAfter(unsafe.Pointer(&partial[0]), p.buf,
				int64(len(partial))*cu.SIZEOF_FLOAT32, &p.boundary)
			p.boundary.Free()
			reduceBuffers <- p.buf
			if p.sum {
				v = float64(combineSum(partial, p.nComp, p.slots))
			} else {
				v = float64(combineMax(partial, p.nComp, p.slots))
			}
		} else if p.sum {
			v = float64(copybackSum(p.buf, p.nComp, p.slots))
		} else {
			v = float64(copybackMax(p.buf, p.nComp, p.slots))
		}
		if p.sqrt {
			v = math.Sqrt(v)
		}
		p.value = v
		p.loaded = true
	}
	return p.value
}

// MaxVecNormAsync enqueues MaxVecNorm without reading the result back.
func MaxVecNormAsync(v *data.Slice) *Pending {
	out := reduceBuf(0, 1, reduceMaxSlots)
	k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), out, 0, v.Len(), reducemaxcfg)
	return &Pending{buf: out, nComp: 1, slots: reduceMaxSlots, sqrt: true}
}

// MaxVecDiffAsync enqueues MaxVecDiff without reading the result back.
func MaxVecDiffAsync(x, y *data.Slice) *Pending {
	util.Argument(x.Len() == y.Len())
	out := reduceBuf(0, 1, reduceMaxSlots)
	k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducemaxcfg)
	return &Pending{buf: out, nComp: 1, slots: reduceMaxSlots, sqrt: true}
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//
//	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	return MaxVecDiffAsync(x, y).Value()
}

var reduceBuffers chan unsafe.Pointer // pool of reduceSlots-wide buffers for reduce

// slot returns the address of the block of reduceSlots floats owned by
// component c of a multi-component reduction.
func slot(buf unsafe.Pointer, c, slots int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(buf) + uintptr(c*slots)*cu.SIZEOF_FLOAT32)
}

// return a reduction buffer from a pool, with nComp*reduceSlots partial slots
// initialized to initVal
func reduceBuf(initVal float32, nComp, slots int) unsafe.Pointer {
	if reduceBuffers == nil {
		initReduceBuf()
	}
	buf := <-reduceBuffers
	cu.MemsetD32Async(cu.DevicePtr(uintptr(buf)), math.Float32bits(initVal),
		int64(nComp*slots), stream0)
	return buf
}

// copybackSum copies the partial results back and adds them in fixed index
// order, then recycles the buffer.
func copybackSum(buf unsafe.Pointer, nComp, slots int) float32 {
	return combineSum(copybackPartials(buf, nComp, slots), nComp, slots)
}

// combineSum adds the partials in fixed index order. Both readback paths call
// this, so a targeted read cannot change the last bits of a sum.
func combineSum(partial []float32, nComp, slots int) float32 {
	var result float32
	for _, value := range partial {
		result += value
	}
	return result
}

// copybackMax copies the partial results back and takes their maximum, then
// recycles the buffer. Maximum is order independent in floating point, so this
// matches the CUDA reduction exactly.
func copybackMax(buf unsafe.Pointer, nComp, slots int) float32 {
	return combineMax(copybackPartials(buf, nComp, slots), nComp, slots)
}

// combineMax takes the maximum of the partials. Shared by both readback paths.
func combineMax(partial []float32, nComp, slots int) float32 {
	result := partial[0]
	for _, value := range partial[1:] {
		if value > result {
			result = value
		}
	}
	return result
}

func copybackPartials(buf unsafe.Pointer, nComp, slots int) []float32 {
	partial := make([]float32, nComp*slots)
	MemCpyDtoH(unsafe.Pointer(&partial[0]), buf,
		int64(len(partial))*cu.SIZEOF_FLOAT32)
	reduceBuffers <- buf
	return partial
}

// initialize pool of reduction buffers. Each holds one block of reduceSlots
// partial results per component of the widest reduction (3).
func initReduceBuf() {
	const N = 128
	reduceBuffers = make(chan unsafe.Pointer, N)
	for i := 0; i < N; i++ {
		reduceBuffers <- MemAlloc(3 * reduceMaxSlots * cu.SIZEOF_FLOAT32)
	}
}

// Launch configurations for the reduce kernels. Grid.X must equal the slot
// count of the matching family: the kernels write one partial per threadgroup
// at index blockIdx.x.
var (
	reducemaxcfg  = &config{Grid: cu.Dim3{X: reduceMaxSlots, Y: 1, Z: 1}, Block: cu.Dim3{X: REDUCE_BLOCKSIZE, Y: 1, Z: 1}}
	reducesumcfg  = &config{Grid: cu.Dim3{X: reduceSumSlots, Y: 1, Z: 1}, Block: cu.Dim3{X: REDUCE_BLOCKSIZE, Y: 1, Z: 1}}
	reducePeakCfg = &config{Grid: cu.Dim3{X: 1, Y: 1, Z: 1}, Block: cu.Dim3{X: reduceMaxSlots, Y: 1, Z: 1}}
)
