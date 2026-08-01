//go:build darwin && arm64 && cgo && go1.24

package metal

import "testing"

func TestCachedKernelEmptyLaunchDoesNotAllocate(t *testing.T) {
	requireTestLibrary(t)

	kernel := NewKernel("mumax3_runtime_test_madd2")
	cfg := Grid(0, 1, 1, 1, 1, 1)
	args := []Arg{
		BufferArg(nil),
		BufferArg(nil),
		F32(1),
		I32(0),
	}
	if err := kernel.Launch(cfg, args...); err != nil {
		t.Fatalf("resolve cached kernel: %v", err)
	}

	var launchErr error
	allocs := testing.AllocsPerRun(1000, func() {
		launchErr = kernel.Launch(cfg, args...)
	})
	if launchErr != nil {
		t.Fatalf("cached empty launch: %v", launchErr)
	}
	if allocs != 0 {
		t.Fatalf("cached empty launch allocations = %g, want 0", allocs)
	}
}
