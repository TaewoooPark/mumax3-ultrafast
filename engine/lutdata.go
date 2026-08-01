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
	ring    []lutSlot          // upload slots, see rotate
	ringPos int

	retiring  [lutRetireGroup]int
	retiringN int
	// Retirement state is fixed per slot group. Keeping it inline avoids a Go
	// allocation on every group rollover in time-dependent runs.
	retirements [lutRing / lutRetireGroup]lutRetirement

	// Queries such as isZero and IsUniform sit in the torque hot path. The
	// table only changes through its source updater, so compute these summaries
	// once per CPU-table version instead of rescanning all 256 regions for every
	// torque evaluation.
	summaryOK      bool
	allZero        bool
	hasZeroValue   bool
	uniformRegions bool
}

type lutSlot struct {
	base    unsafe.Pointer
	ptrs    cuda.LUTPtrs
	retired *lutRetirement
}

type lutRetirement struct {
	completion cuda.Completion
	refs       int
	complete   bool
}

type updater interface {
	update() // updates cpu lookup table
}

func (p *lut) init(nComp int, source updater) {
	p.free()
	p.gpu_buf = make(cuda.LUTPtrs, nComp)
	p.gpu_ok = false
	p.cpu_buf = make([][NREGION]float32, nComp)
	p.source = source
	p.summaryOK = false
	p.allZero = false
	p.hasZeroValue = false
	p.uniformRegions = false
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

// Grouping retirement points avoids a cgo call and an Objective-C retain on
// every solver stage. The token is recorded after every reader in the group,
// so it safely covers all of them. Sixteen divides the 64-slot ring and still
// leaves at least 48 later uploads for the GPU to complete before reuse.
const lutRetireGroup = 16

// get an up-to-date version of the lookup-table on GPU
func (p *lut) gpuLUT() cuda.LUTPtrs {
	p.source.update()
	if !p.gpu_ok {
		// rotate hands back a table no encoded kernel can still be reading.
		// Completion-aware retirement removes the full pipeline drain that the
		// previous ring needed on every 64th upload.
		p.rotate()
		// [][NREGION]float32 stores its fixed-size component arrays back to
		// back, matching the one-allocation GPU slot. One host copy therefore
		// updates every vector component and avoids three runtime crossings.
		cuda.MemCpyHtoDUnordered(
			p.ring[p.ringPos].base,
			unsafe.Pointer(&p.cpu_buf[0][0]),
			int64(len(p.cpu_buf))*NREGION*cu.SIZEOF_FLOAT32,
		)
		p.gpu_ok = true
	}
	return p.gpu_buf
}

// rotate points gpu_buf at a slot that no in-flight kernel reads. When a slot
// group closes, its completion only retains the current ordered-queue tail; it
// neither submits nor waits. Every reader of that group has already been
// encoded by then, so the tail safely retires all of its slots.
//
// Slots are allocated on demand, so a static table keeps one small allocation.
// At wrap, a completed candidate is reused immediately. Otherwise Wait targets
// only the command buffer captured for that slot. This remains safe across an
// MPS commitAndContinue: the token retains the already-committed old root while
// the runtime owns the replacement, so no stale command buffer is recommitted.
func (p *lut) rotate() {
	if len(p.ring) != 0 {
		p.retireCurrent()
	}

	switch {
	case len(p.ring) < lutRing:
		p.ring = append(p.ring, newLUTSlot(len(p.cpu_buf)))
		p.ringPos = len(p.ring) - 1
	case p.ringPos+1 < lutRing:
		p.ringPos++
	default:
		p.ringPos = 0
	}

	candidate := &p.ring[p.ringPos]
	p.reuse(candidate)
	p.gpu_buf = candidate.ptrs
}

func (p *lut) retireCurrent() {
	current := &p.ring[p.ringPos]
	util.Assert(current.retired == nil)
	p.retiring[p.retiringN] = p.ringPos
	p.retiringN++
	if p.retiringN != lutRetireGroup {
		return
	}

	completion := cuda.RecordCompletion()
	if completion.Valid() {
		groupStart := p.retiring[0]
		util.Assert(groupStart%lutRetireGroup == 0)
		retirement := &p.retirements[groupStart/lutRetireGroup]
		util.Assert(retirement.refs == 0)
		*retirement = lutRetirement{
			completion: completion,
			refs:       p.retiringN,
		}
		for i := 0; i < p.retiringN; i++ {
			slot := &p.ring[p.retiring[i]]
			util.Assert(p.retiring[i] == groupStart+i)
			util.Assert(slot.retired == nil)
			slot.retired = retirement
		}
	}
	p.retiringN = 0
}

func (p *lut) reuse(slot *lutSlot) {
	retirement := slot.retired
	if retirement == nil {
		return
	}
	if !retirement.complete {
		if !retirement.completion.Ready() {
			retirement.completion.Wait()
		}
		retirement.completion.Free()
		retirement.complete = true
	}
	p.releaseRetirement(slot)
}

func (p *lut) releaseRetirement(slot *lutSlot) {
	retirement := slot.retired
	if retirement == nil {
		return
	}
	slot.retired = nil
	retirement.refs--
	util.Assert(retirement.refs >= 0)
	if retirement.refs == 0 {
		retirement.completion.Free()
		*retirement = lutRetirement{}
	}
}

func newLUTSlot(nComp int) lutSlot {
	componentBytes := uintptr(NREGION * cu.SIZEOF_FLOAT32)
	base := cuda.MemAlloc(int64(componentBytes) * int64(nComp))
	ptrs := make(cuda.LUTPtrs, nComp)
	for c := range ptrs {
		ptrs[c] = unsafe.Add(base, uintptr(c)*componentBytes)
	}
	return lutSlot{base: base, ptrs: ptrs}
}

// free releases a LUT ring deterministically. Parameters normally live for
// the process lifetime; the method also makes reinitialization and focused
// tests leak-free. MemFree supplies the final GPU-use synchronization.
func (p *lut) free() {
	for i := range p.ring {
		p.releaseRetirement(&p.ring[i])
		cuda.MemFree(p.ring[i].base)
		p.ring[i] = lutSlot{}
	}
	p.ring = nil
	p.ringPos = 0
	p.retiring = [lutRetireGroup]int{}
	p.retiringN = 0
	p.retirements = [lutRing / lutRetireGroup]lutRetirement{}
	p.gpu_buf = nil
	p.gpu_ok = false
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
