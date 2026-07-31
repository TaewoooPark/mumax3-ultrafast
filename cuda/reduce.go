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
	util.Argument(in.NComp() == 1)
	out := reduceBuf(0, 1, reduceSumSlots)
	k_reducesum_async(in.DevPtr(0), out, 0, in.Len(), reducesumcfg)
	return copybackSum(out, 1, reduceSumSlots)
}

// Dot product.
func Dot(a, b *data.Slice) float32 {
	nComp := a.NComp()
	util.Argument(nComp == b.NComp())
	out := reduceBuf(0, nComp, reduceSumSlots)
	// Not async over components. Each component writes its own block of slots
	// rather than accumulating into a shared one, so the sum stays independent
	// of both threadgroup scheduling and inter-launch overlap.
	for c := 0; c < nComp; c++ {
		k_reducedot_async(a.DevPtr(c), b.DevPtr(c), slot(out, c, reduceSumSlots), 0, a.Len(), reducesumcfg)
	}
	return copybackSum(out, nComp, reduceSumSlots)
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
	out := reduceBuf(0, 1, reduceMaxSlots)
	k_reducemaxvecnorm2_async(v.DevPtr(0), v.DevPtr(1), v.DevPtr(2), out, 0, v.Len(), reducemaxcfg)
	return math.Sqrt(float64(copybackMax(out, 1, reduceMaxSlots)))
}

// Maximum of the norms of the difference between all vectors (x1,y1,z1) and (x2,y2,z2)
//
//	(dx, dy, dz) = (x1, y1, z1) - (x2, y2, z2)
//	max_i sqrt( dx[i]*dx[i] + dy[i]*dy[i] + dz[i]*dz[i] )
func MaxVecDiff(x, y *data.Slice) float64 {
	util.Argument(x.Len() == y.Len())
	out := reduceBuf(0, 1, reduceMaxSlots)
	k_reducemaxvecdiff2_async(x.DevPtr(0), x.DevPtr(1), x.DevPtr(2),
		y.DevPtr(0), y.DevPtr(1), y.DevPtr(2),
		out, 0, x.Len(), reducemaxcfg)
	return math.Sqrt(float64(copybackMax(out, 1, reduceMaxSlots)))
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
	partial := copybackPartials(buf, nComp, slots)
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
	partial := copybackPartials(buf, nComp, slots)
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
	reducemaxcfg = &config{Grid: cu.Dim3{X: reduceMaxSlots, Y: 1, Z: 1}, Block: cu.Dim3{X: REDUCE_BLOCKSIZE, Y: 1, Z: 1}}
	reducesumcfg = &config{Grid: cu.Dim3{X: reduceSumSlots, Y: 1, Z: 1}, Block: cu.Dim3{X: REDUCE_BLOCKSIZE, Y: 1, Z: 1}}
)
