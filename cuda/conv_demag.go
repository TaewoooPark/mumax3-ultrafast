package cuda

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Stores the necessary state to perform FFT-accelerated convolution
// with magnetostatic kernel (or other kernel of same symmetry).
type DemagConvolution struct {
	inputSize        [3]int            // 3D size of the input/output data
	realKernSize     [3]int            // Size of kernel and logical FFT size.
	fftKernLogicSize [3]int            // logic size FFTed kernel, real parts only, we store less
	fftRBuf          [3]*data.Slice    // FFT input buf; 2D: Z shares storage with X.
	fftCBuf          [3]*data.Slice    // FFT output buf; 2D: Z shares storage with X.
	fftBwBuf         *data.Slice       // inverse FFT output, see bwFFT
	kern             [3][3]*data.Slice // FFT kernel on device
	fwPlan           fft3DR2CPlan      // Forward FFT (1 component)
	bwPlan           fft3DC2RPlan      // Backward FFT (1 component)
}

// Initializes a convolution to evaluate the demag field for the given mesh geometry.
// Sanity-checked if test == true (slow-ish for large meshes).
func NewDemag(inputSize, PBC [3]int, kernel [3][3]*data.Slice, test bool) *DemagConvolution {
	c := new(DemagConvolution)
	c.inputSize = inputSize
	c.realKernSize = kernel[X][X].Size()
	c.init(kernel)
	if test {
		testConvolution(c, PBC, kernel)
	}
	return c
}

// Calculate the demag field of m * vol * Msat, store result in B.
//
//	m:    magnetization normalized to unit length
//	vol:  unitless mask used to scale m's length, may be nil
//	Msat: saturation magnetization in A/m
//	B:    resulting demag field, in Tesla
func (c *DemagConvolution) Exec(B, m, vol *data.Slice, Msat MSlice) {
	util.Argument(B.Size() == c.inputSize && m.Size() == c.inputSize)
	if c.is2D() {
		c.exec2D(B, m, vol, Msat)
	} else {
		c.exec3D(B, m, vol, Msat)
	}
}

func (c *DemagConvolution) exec3D(outp, inp, vol *data.Slice, Msat MSlice) {
	for i := 0; i < 3; i++ { // FW FFT
		c.fwFFT(i, inp, vol, Msat)
	}

	// kern mul
	kernMulRSymm3D_async(c.fftCBuf,
		c.kern[X][X], c.kern[Y][Y], c.kern[Z][Z],
		c.kern[Y][Z], c.kern[X][Z], c.kern[X][Y],
		c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	for i := 0; i < 3; i++ { // BW FFT
		c.bwFFT(i, outp)
	}
}

func (c *DemagConvolution) exec2D(outp, inp, vol *data.Slice, Msat MSlice) {
	// Convolution is separated into
	// a 1D convolution for z and a 2D convolution for xy.
	// So only 2 FFT buffers are needed at the same time.
	Nx, Ny := c.fftKernLogicSize[X], c.fftKernLogicSize[Y]

	// Z
	c.fwFFT(Z, inp, vol, Msat)
	kernMulRSymm2Dz_async(c.fftCBuf[Z], c.kern[Z][Z], Nx, Ny)
	c.bwFFT(Z, outp)

	// XY
	c.fwFFT(X, inp, vol, Msat)
	c.fwFFT(Y, inp, vol, Msat)
	kernMulRSymm2Dxy_async(c.fftCBuf[X], c.fftCBuf[Y],
		c.kern[X][X], c.kern[Y][Y], c.kern[X][Y], Nx, Ny)
	c.bwFFT(X, outp)
	c.bwFFT(Y, outp)
}

func (c *DemagConvolution) is2D() bool {
	return c.inputSize[Z] == 1
}

// zero 1-component slice
func zero1_async(dst *data.Slice) {
	cu.MemsetD32Async(cu.DevicePtr(uintptr(dst.DevPtr(0))), 0, int64(dst.Len()), stream0)
}

// forward FFT component i
//
// copyPadMul writes exactly the inputSize corner of the padded buffer and
// leaves the rest untouched, so the zero padding only has to be written once.
// The inverse transform is what used to destroy it, by writing the whole padded
// buffer; it now targets fftBwBuf instead. That removes one full padded-buffer
// clear plus one dispatch per component per field evaluation, which is 18
// dispatches per RK45DP step.
func (c *DemagConvolution) fwFFT(i int, inp, vol *data.Slice, Msat MSlice) {
	in := inp.Comp(i)
	copyPadMul(c.fftRBuf[i], in, vol, c.realKernSize, c.inputSize, Msat)
	c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
}

// backward FFT component i
//
// All components share one output buffer: each inverse transform is fully
// consumed by its copyUnPad before the next one is encoded on the same ordered
// queue.
func (c *DemagConvolution) bwFFT(i int, outp *data.Slice) {
	c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftBwBuf)
	out := outp.Comp(i)
	copyUnPad(out, c.fftBwBuf, c.inputSize, c.realKernSize)
}

func (c *DemagConvolution) init(realKern [3][3]*data.Slice) {
	// init device buffers
	// 2D re-uses fftBuf[X] as fftBuf[Z], 3D needs all 3 fftBufs.
	nc := fftR2COutputSizeFloats(c.realKernSize)
	c.fftCBuf[X] = NewSlice(1, nc)
	c.fftCBuf[Y] = NewSlice(1, nc)
	if c.is2D() {
		c.fftCBuf[Z] = c.fftCBuf[X]
	} else {
		c.fftCBuf[Z] = NewSlice(1, nc)
	}

	c.fftRBuf[X] = NewSlice(1, c.realKernSize)
	c.fftRBuf[Y] = NewSlice(1, c.realKernSize)
	if c.is2D() {
		c.fftRBuf[Z] = c.fftRBuf[X]
	} else {
		c.fftRBuf[Z] = NewSlice(1, c.realKernSize)
	}

	// Dedicated inverse-FFT output, so the forward buffers keep their zero
	// padding for the whole run. See fwFFT.
	c.fftBwBuf = NewSlice(1, c.realKernSize)

	// init FFT plans
	//
	// The padded buffer only holds data in its inputSize corner; everything
	// else is zero and stays zero (see fwFFT), and copyUnPad only reads that
	// corner back. The transform along X therefore maps the padded Y rows to
	// zero rows on the way in, and only the data rows are needed on the way
	// out, so both X passes can skip them. That is exact, not an
	// approximation: the FFT of a zero row is a zero row.
	//
	// The Y and Z passes must still see every row, because those zeros are the
	// zero-padding that turns the cyclic convolution into a linear one.
	//
	// Only claim it when the data rows are a contiguous prefix of the buffer,
	// which needs a single Z plane. The bridge re-checks this and ignores the
	// hint otherwise.
	activeY := 0
	if c.realKernSize[Z] == 1 {
		activeY = c.inputSize[Y]
	}
	c.fwPlan = newFFT3DR2C(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z], activeY)
	c.bwPlan = newFFT3DC2R(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z], activeY)

	// init FFT kernel

	// logic size of FFT(kernel): store real parts only
	c.fftKernLogicSize = fftR2COutputSizeFloats(c.realKernSize)
	util.Assert(c.fftKernLogicSize[X]%2 == 0)
	c.fftKernLogicSize[X] /= 2

	// physical size of FFT(kernel): store only non-redundant part exploiting Y, Z mirror symmetry
	// X mirror symmetry already exploited: FFT(kernel) is purely real.
	physKSize := [3]int{c.fftKernLogicSize[X], c.fftKernLogicSize[Y]/2 + 1, c.fftKernLogicSize[Z]/2 + 1}

	// The transforms below carry the demag kernel, which fills the whole padded
	// array rather than only the inputSize corner, so they must not use the
	// zero-row hint that c.fwPlan carries. Use a full-extent plan for them and
	// release it once the kernel is in Fourier space.
	kernPlan := newFFT3DR2C(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z], 0)
	defer kernPlan.Free()

	output := c.fftCBuf[0]
	input := c.fftRBuf[0]

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ { // upper triangular part
			if realKern[i][j] != nil { // ignore 0's
				// FW FFT
				data.Copy(input, realKern[i][j])
				kernPlan.ExecAsync(input, output)
				c.kern[i][j] = compressFFTKernel(output, physKSize, 1/float32(kernPlan.InputLen()))
			}
		}
	}

	// The kernel transforms above copied full-size kernels through
	// fftRBuf[0], so restore the zero padding that fwFFT now relies on.
	// NewSlice already zeroes, so only the buffers used here need clearing.
	zero1_async(c.fftRBuf[X])
	zero1_async(c.fftRBuf[Y])
	if !c.is2D() {
		zero1_async(c.fftRBuf[Z])
	}
}

func (c *DemagConvolution) Free() {
	if c == nil {
		return
	}
	c.inputSize = [3]int{}
	c.realKernSize = [3]int{}
	c.fftBwBuf.Free()
	c.fftBwBuf = nil
	for i := 0; i < 3; i++ {
		c.fftCBuf[i].Free()
		c.fftRBuf[i].Free()
		c.fftCBuf[i] = nil
		c.fftRBuf[i] = nil

		for j := 0; j < 3; j++ {
			c.kern[i][j].Free()
			c.kern[i][j] = nil
		}
		c.fwPlan.Free()
		c.bwPlan.Free()

		cudaCtx.SetCurrent()
	}
}
