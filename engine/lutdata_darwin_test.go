//go:build darwin && arm64 && cgo

package engine

import (
	"testing"
	"unsafe"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/metal"
	"github.com/mumax/3/data"
)

type noOpLUTUpdater struct{}

func (noOpLUTUpdater) update() {}

type benchmarkLUTUploader interface {
	upload(iteration int) cuda.LUTPtrs
	free()
}

type completionRingBenchmarkLUT struct {
	table lut
}

func newCompletionRingBenchmarkLUT(components int) *completionRingBenchmarkLUT {
	uploader := &completionRingBenchmarkLUT{}
	uploader.table.init(components, noOpLUTUpdater{})
	return uploader
}

func (uploader *completionRingBenchmarkLUT) upload(
	iteration int,
) cuda.LUTPtrs {
	for component := range uploader.table.cpu_buf {
		uploader.table.cpu_buf[component][0] =
			float32(iteration*len(uploader.table.cpu_buf) + component + 1)
	}
	uploader.table.invalidateCPU()
	return uploader.table.gpuLUT()
}

func (uploader *completionRingBenchmarkLUT) free() {
	uploader.table.free()
}

// legacySyncBenchmarkLUT deliberately mirrors the former production path so
// the benchmark measures the whole LUT change, including its separate
// component allocations/uploads and its full queue drain on every ring wrap.
type legacySyncBenchmarkLUT struct {
	cpuBuf [][NREGION]float32
	ring   []cuda.LUTPtrs
	pos    int
}

func newLegacySyncBenchmarkLUT(components int) *legacySyncBenchmarkLUT {
	return &legacySyncBenchmarkLUT{
		cpuBuf: make([][NREGION]float32, components),
	}
}

func (uploader *legacySyncBenchmarkLUT) upload(
	iteration int,
) cuda.LUTPtrs {
	for component := range uploader.cpuBuf {
		uploader.cpuBuf[component][0] =
			float32(iteration*len(uploader.cpuBuf) + component + 1)
	}

	switch {
	case len(uploader.ring) < lutRing:
		slot := make(cuda.LUTPtrs, len(uploader.cpuBuf))
		for component := range slot {
			slot[component] = cuda.MemAlloc(NREGION * data.SIZEOF_FLOAT32)
		}
		uploader.ring = append(uploader.ring, slot)
		uploader.pos = len(uploader.ring) - 1
	case uploader.pos+1 < lutRing:
		uploader.pos++
	default:
		cuda.Sync()
		uploader.pos = 0
	}

	result := uploader.ring[uploader.pos]
	for component := range result {
		cuda.MemCpyHtoDUnordered(
			result[component],
			unsafe.Pointer(&uploader.cpuBuf[component][0]),
			NREGION*data.SIZEOF_FLOAT32,
		)
	}
	return result
}

func (uploader *legacySyncBenchmarkLUT) free() {
	for _, slot := range uploader.ring {
		for _, pointer := range slot {
			cuda.MemFree(pointer)
		}
	}
	uploader.ring = nil
}

func TestLUTRingCompletionRetirementAndVectorOrdering(t *testing.T) {
	cuda.Init(0)
	const (
		components = 3
		uploads    = 300
	)

	var table lut
	table.init(components, noOpLUTUpdater{})
	defer table.free()
	regionMap := cuda.NewBytes(1) // region zero
	defer regionMap.Free()
	output := cuda.NewSlice(components, [3]int{uploads, 1, 1})
	defer output.Free()
	if err := metal.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := metal.ResetRuntimeStats(); err != nil {
		t.Fatal(err)
	}

	for upload := 0; upload < uploads; upload++ {
		for component := 0; component < components; component++ {
			table.cpu_buf[component][0] =
				float32(upload*components + component + 1)
		}
		table.invalidateCPU()
		gpuTable := table.gpuLUT()
		for component := 0; component < components; component++ {
			pointer := unsafe.Add(
				output.DevPtr(component),
				upload*data.SIZEOF_FLOAT32,
			)
			view := data.SliceFromPtrs(
				[3]int{1, 1, 1},
				data.GPUMemory,
				[]unsafe.Pointer{pointer},
			)
			cuda.RegionDecode(
				view,
				cuda.LUTPtr(gpuTable[component]),
				regionMap,
			)
			view.Disable()
		}
	}

	stats, err := metal.GetRuntimeStats()
	if err != nil {
		t.Fatal(err)
	}
	if stats.FullDrains != 0 {
		t.Fatalf("300 LUT uploads caused %d global drains", stats.FullDrains)
	}
	wantRecords := (uploads - 1) / lutRetireGroup
	if stats.CompletionRecords != uint64(wantRecords) {
		t.Fatalf(
			"completion records = %d, want %d",
			stats.CompletionRecords,
			wantRecords,
		)
	}
	if table.retiringN != (uploads-1)%lutRetireGroup {
		t.Fatalf(
			"pending retirement slots = %d, want %d",
			table.retiringN,
			(uploads-1)%lutRetireGroup,
		)
	}
	if stats.CommandBufferSubmissions >= uploads/8 {
		t.Fatalf(
			"%d command-buffer submissions for %d uploads suggests per-upload submission",
			stats.CommandBufferSubmissions,
			uploads,
		)
	}
	t.Logf(
		"uploads=%d submissions=%d records=%d queries=%d hits=%d targeted-waits=%d wait-submissions=%d full-drains=%d",
		uploads,
		stats.CommandBufferSubmissions,
		stats.CompletionRecords,
		stats.CompletionQueries,
		stats.CompletionQueryHits,
		stats.CompletionWaits,
		stats.CompletionWaitSubmissions,
		stats.FullDrains,
	)
	if len(table.ring) != lutRing {
		t.Fatalf("ring slots = %d, want %d", len(table.ring), lutRing)
	}
	componentBytes := uintptr(NREGION * data.SIZEOF_FLOAT32)
	for slotIndex := range table.ring {
		slot := &table.ring[slotIndex]
		for component := range slot.ptrs {
			want := unsafe.Add(slot.base, uintptr(component)*componentBytes)
			if slot.ptrs[component] != want {
				t.Fatalf(
					"slot %d component %d pointer is not contiguous",
					slotIndex,
					component,
				)
			}
		}
	}

	host := output.HostCopy()
	defer host.Free()
	values := host.Host()
	for upload := 0; upload < uploads; upload++ {
		for component := 0; component < components; component++ {
			want := float32(upload*components + component + 1)
			if got := values[component][upload]; got != want {
				t.Fatalf(
					"upload %d component %d = %g, want %g",
					upload,
					component,
					got,
					want,
				)
			}
		}
	}
	table.summaryOK = true
	table.allZero = true
	table.hasZeroValue = true
	table.uniformRegions = true
	table.init(1, noOpLUTUpdater{})
	if len(table.ring) != 0 || len(table.gpu_buf) != 1 {
		t.Fatal("reinitialization did not replace LUT GPU state")
	}
	if table.summaryOK || table.allZero || table.hasZeroValue ||
		table.uniformRegions {
		t.Fatal("reinitialization did not clear LUT summaries")
	}
	table.free()
	if len(table.ring) != 0 || table.gpu_buf != nil {
		t.Fatal("free did not clear LUT ring state")
	}
}

func BenchmarkTimeDependentLUTUpload(b *testing.B) {
	cuda.Init(0)
	const (
		components = 3
		cells      = NREGION
	)

	benchmarks := []struct {
		name string
		new  func(int) benchmarkLUTUploader
	}{
		{"completion-ring", func(n int) benchmarkLUTUploader {
			return newCompletionRingBenchmarkLUT(n)
		}},
		{"legacy-global-drain", func(n int) benchmarkLUTUploader {
			return newLegacySyncBenchmarkLUT(n)
		}},
	}
	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			uploader := benchmark.new(components)
			defer uploader.free()
			regionMap := cuda.NewBytes(cells)
			defer regionMap.Free()
			output := cuda.NewSlice(components, [3]int{cells, 1, 1})
			defer output.Free()
			componentOutput := make([]*data.Slice, components)
			for component := range componentOutput {
				componentOutput[component] = output.Comp(component)
				defer componentOutput[component].Disable()
			}

			if err := metal.Sync(); err != nil {
				b.Fatal(err)
			}
			if err := metal.ResetRuntimeStats(); err != nil {
				b.Fatal(err)
			}
			b.SetBytes(components * NREGION * data.SIZEOF_FLOAT32)
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				gpuTable := uploader.upload(iteration)
				for component := range componentOutput {
					cuda.RegionDecode(
						componentOutput[component],
						cuda.LUTPtr(gpuTable[component]),
						regionMap,
					)
				}
			}
			// Include one terminal fence so the result is completed GPU work,
			// rather than merely CPU-side command encoding throughput.
			cuda.Sync()
			b.StopTimer()

			stats, err := metal.GetRuntimeStats()
			if err != nil {
				b.Fatal(err)
			}
			hotDrains := stats.FullDrains
			if hotDrains != 0 {
				hotDrains-- // exclude the benchmark's one terminal fence
			}
			if b.N != 0 {
				operations := float64(b.N)
				b.ReportMetric(float64(hotDrains)/operations, "hot-drain/op")
				b.ReportMetric(
					float64(stats.CompletionRecords)/operations,
					"record/op",
				)
				b.ReportMetric(
					float64(stats.CompletionWaits)/operations,
					"target-wait/op",
				)
			}
		})
	}
}
