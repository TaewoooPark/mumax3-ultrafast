package cuda

import (
	"fmt"
	"github.com/mumax/3/cuda/cu"
)

// CUDA Launch parameters.
// there might be better choices for recent hardware,
// but it barely makes a difference in the end.
const MaxGridSize = 65535

// cuda launch configuration
//
// Grid and Block carry the CUDA geometry: Grid*Block threads are dispatched and
// the kernel guards against the round-up. Threads optionally requests an exact
// thread count instead, which lets Metal form a partial trailing threadgroup.
// A kernel may only be launched that way if it indexes
// thread_position_in_grid: for a partial group Metal reports the reduced size in
// threads_per_threadgroup, which would corrupt the CUDA blockDim arithmetic.
// Zero means "use Grid*Block", which is what the reduction kernels require
// because their grid-stride loop strides by gridDim.x*blockDim.x.
type config struct {
	Grid, Block cu.Dim3
	Threads     cu.Dim3
}

// Make a 1D kernel launch configuration suited for N threads.
func make1DConf(N int) *config {
	bl := cu.Dim3{X: BlockSize, Y: 1, Z: 1}

	n2 := divUp(N, BlockSize) // N2 blocks left
	nx := divUp(n2, MaxGridSize)
	ny := divUp(n2, nx)
	gr := cu.Dim3{X: nx, Y: ny, Z: 1}

	return &config{Grid: gr, Block: bl}
}

// Make a 3D kernel launch configuration suited for N threads.
func make3DConf(N [3]int) *config {
	bl := cu.Dim3{X: TileX, Y: TileY, Z: 1}

	nx := divUp(N[X], TileX)
	ny := divUp(N[Y], TileY)
	gr := cu.Dim3{X: nx, Y: ny, Z: N[Z]}

	return &config{Grid: gr, Block: bl}
}

// integer minimum
func iMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Integer division rounded up.
func divUp(x, y int) int {
	return ((x - 1) / y) + 1
}

const (
	X = 0
	Y = 1
	Z = 2
)

func checkSize(a interface {
	Size() [3]int
}, b ...interface {
	Size() [3]int
}) {
	sa := a.Size()
	for _, b := range b {
		if b.Size() != sa {
			panic(fmt.Sprintf("size mismatch: %v != %v", sa, b.Size()))
		}
	}
}
