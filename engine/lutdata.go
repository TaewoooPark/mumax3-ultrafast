package engine

import (
	"unsafe"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// look-up table for region based parameters
type lut struct {
	gpu_buf cuda.LUTPtrs       // gpu copy of cpu buffer, only transferred when needed
	gpu_ok  bool               // gpu cache up-to date with cpu source?
	cpu_buf [][NREGION]float32 // table data on cpu
	source  updater            // updates cpu data
	ring    []cuda.LUTPtrs     // upload slots, see rotate
	ringPos int

	// Queries such as isZero and IsUniform sit in the torque hot path. The
	// table only changes through its source updater, so compute these summaries
	// once per CPU-table version instead of rescanning all 256 regions for every
	// torque evaluation.
	summaryOK      bool
	allZero        bool
	hasZeroValue   bool
	uniformRegions bool
}

type updater interface {
	update() // updates cpu lookup table
}

func (p *lut) init(nComp int, source updater) {
	p.gpu_buf = make(cuda.LUTPtrs, nComp)
	p.cpu_buf = make([][NREGION]float32, nComp)
	p.source = source
}

// invalidateCPU marks every value derived from cpu_buf stale. Sources call
// this after changing the table; read-only GPU slot rotation does not affect
// the summaries.
func (p *lut) invalidateCPU() {
	p.gpu_ok = false
	p.summaryOK = false
}

// get an up-to-date version of the lookup-table on CPU
func (p *lut) cpuLUT() [][NREGION]float32 {
	p.source.update()
	return p.cpu_buf
}

// Number of upload slots a table rotates through. A table that never changes
// only ever allocates the first one.
const lutRing = 64

// get an up-to-date version of the lookup-table on GPU
func (p *lut) gpuLUT() cuda.LUTPtrs {
	p.source.update()
	if !p.gpu_ok {
		// Upload to GPU. rotate hands back a table no encoded kernel can still
		// be reading, which is what used to require draining the pipeline
		// before and after the 1 KB write. A parameter that is a function of
		// time is re-uploaded on every solver stage, so those two drains were
		// the dominant cost of the whole step on Metal.
		p.rotate()
		for c := range p.gpu_buf {
			cuda.MemCpyHtoDUnordered(p.gpu_buf[c], unsafe.Pointer(&p.cpu_buf[c][0]), cu.SIZEOF_FLOAT32*NREGION)
		}
		p.gpu_ok = true
	}
	return p.gpu_buf
}

// rotate points gpu_buf at a slot that no in-flight kernel reads.
//
// Slots are allocated on demand, so a static table keeps a single one. Once
// the ring is full it is reused from the start, and only that wrap needs a
// synchronization: every slot handed out after it was last written before the
// wrap. That turns two drains per upload into one drain per lutRing uploads.
func (p *lut) rotate() {
	switch {
	case len(p.ring) < lutRing:
		slot := make(cuda.LUTPtrs, len(p.cpu_buf))
		for c := range slot {
			slot[c] = cuda.MemAlloc(NREGION * cu.SIZEOF_FLOAT32)
		}
		p.ring = append(p.ring, slot)
		p.ringPos = len(p.ring) - 1
	case p.ringPos+1 < lutRing:
		p.ringPos++
	default:
		cuda.Sync() // the ring wrapped: retire every reader of every slot
		p.ringPos = 0
	}
	p.gpu_buf = p.ring[p.ringPos]
}

// utility for LUT of single-component data
func (p *lut) gpuLUT1() cuda.LUTPtr {
	util.Assert(len(p.gpu_buf) == 1)
	return cuda.LUTPtr(p.gpuLUT()[0])
}

// all data is 0?
func (p *lut) isZero() bool {
	p.updateSummary()
	return p.allZero
}

func (p *lut) nonZero() bool { return !p.isZero() }

// some data is 0?
func (p *lut) hasZero() bool {
	p.updateSummary()
	return p.hasZeroValue
}

func (p *lut) isUniform() bool {
	p.updateSummary()
	return p.uniformRegions
}

func (p *lut) updateSummary() {
	v := p.cpuLUT()
	if p.summaryOK {
		return
	}
	p.allZero = true
	p.hasZeroValue = false
	p.uniformRegions = true
	for c := range v {
		for i := 0; i < NREGION; i++ {
			value := v[c][i]
			if value == 0 {
				p.hasZeroValue = true
			} else {
				p.allZero = false
			}
			// Preserve the old NaN semantics: NaN != NaN makes a table
			// containing NaNs non-uniform, even if every region is NaN.
			if i != 0 && value != v[c][0] {
				p.uniformRegions = false
			}
		}
	}
	p.summaryOK = true
}

func (b *lut) NComp() int { return len(b.cpu_buf) }

// uncompress the table to a full array with parameter values per cell.
func (p *lut) Slice() (*data.Slice, bool) {
	b := cuda.Buffer(p.NComp(), Mesh().Size())
	p.EvalTo(b)
	return b, true
}

// uncompress the table to a full array in the dst Slice with parameter values per cell.
func (p *lut) EvalTo(dst *data.Slice) {
	gpu := p.gpuLUT()
	for c := 0; c < p.NComp(); c++ {
		cuda.RegionDecode(dst.Comp(c), cuda.LUTPtr(gpu[c]), regions.Gpu())
	}
}
