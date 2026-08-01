package data

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestIndex(t *testing.T) {
	mesh := [3]int{6, 5, 4}
	slice := NewSlice(7, mesh)
	data := slice.Tensors()

	if len(data) != 7 { //c
		t.Fail()
	}
	if len(data[0]) != 4 { // z
		t.Fail()
	}
	if len(data[0][0]) != 5 { // y
		t.Fail()
	}
	if len(data[0][0][0]) != 6 { // x
		t.Fail()
	}

	slice.Set(4, 5, 4, 3, 345) // c x y z
	if data[4][3][4][5] != 345 {
		t.Fail()
	}
}

func TestCPUComponentViewIsNonOwning(t *testing.T) {
	mesh := [3]int{4, 2, 1}
	owner := NewSlice(3, mesh)
	if !owner.OwnsStorage() {
		t.Fatal("new CPU slice should own its storage")
	}

	for c := range owner.Host() {
		for i := range owner.Host()[c] {
			owner.Host()[c][i] = float32(100*c + i)
		}
	}

	view := owner.Comp(1)
	if view.OwnsStorage() {
		t.Fatal("component view unexpectedly owns parent storage")
	}
	if !view.Contiguous() {
		t.Fatal("a one-component view must be contiguous")
	}
	if &view.ptrs[0] != &owner.ptrs[1] {
		t.Fatal("Comp allocated a separate component pointer table")
	}
	view.Host()[0][3] = 1234
	if got := owner.Host()[1][3]; got != 1234 {
		t.Fatalf("component write did not reach parent: got %v", got)
	}

	copyOfView := NewSlice(1, mesh)
	Copy(copyOfView, view)
	if got := copyOfView.Host()[0][3]; got != 1234 {
		t.Fatalf("component copy = %v, want 1234", got)
	}
	copyOfView.Free()

	view.Free()
	view.Free() // freeing a view is idempotent
	if view.NComp() != 0 || view.OwnsStorage() {
		t.Fatal("freed view was not disabled")
	}
	if !owner.OwnsStorage() || owner.NComp() != 3 {
		t.Fatal("freeing a view disabled its parent")
	}
	if got := owner.Host()[1][3]; got != 1234 {
		t.Fatalf("freeing a view damaged parent storage: got %v", got)
	}

	owner.Free()
	owner.Free() // owner double-free is also safe
	if owner.NComp() != 0 || owner.OwnsStorage() {
		t.Fatal("freed owner was not disabled")
	}
}

func TestCPUVectorSliceIsContiguous(t *testing.T) {
	size := [3]int{3, 2, 1}
	s := NewSlice(3, size)
	defer s.Free()
	if !s.Contiguous() {
		t.Fatal("NewSlice should use one contiguous host allocation")
	}
	stride := uintptr(s.Len()) * SIZEOF_FLOAT32
	if uintptr(s.ptrs[1])-uintptr(s.ptrs[0]) != stride ||
		uintptr(s.ptrs[2])-uintptr(s.ptrs[1]) != stride {
		t.Fatal("host component pointers do not have the expected stride")
	}
	for c := range s.Host() {
		for i := range s.Host()[c] {
			s.Host()[c][i] = float32(100*c + i)
		}
	}
	copySlice := NewSlice(3, size)
	defer copySlice.Free()
	Copy(copySlice, s)
	for c := range s.Host() {
		for i, want := range s.Host()[c] {
			if got := copySlice.Host()[c][i]; got != want {
				t.Fatalf("copy component %d element %d = %v, want %v", c, i, got, want)
			}
		}
	}
}

func TestContiguousGPUCopyUsesOneTransfer(t *testing.T) {
	oldFree, oldFreeHost := memFree, memFreeHost
	oldCopy, oldDtoH, oldHtoD := memCpy, memCpyDtoH, memCpyHtoD
	defer func() {
		memFree, memFreeHost = oldFree, oldFreeHost
		memCpy, memCpyDtoH, memCpyHtoD = oldCopy, oldDtoH, oldHtoD
	}()

	copyBytes := func(dst, src unsafe.Pointer, bytes int64) {
		copy(unsafe.Slice((*byte)(dst), int(bytes)), unsafe.Slice((*byte)(src), int(bytes)))
	}
	dtoHCalls := 0
	EnableGPU(
		func(unsafe.Pointer) {},
		func(unsafe.Pointer) {},
		copyBytes,
		func(dst, src unsafe.Pointer, bytes int64) {
			dtoHCalls++
			copyBytes(dst, src, bytes)
		},
		copyBytes,
	)

	size := [3]int{4, 2, 1}
	length := prod(size)
	backing := make([]float32, 3*length)
	for i := range backing {
		backing[i] = float32(i + 1)
	}
	base := unsafe.Pointer(unsafe.SliceData(backing))
	stride := uintptr(length) * SIZEOF_FLOAT32
	src := SliceFromContiguousPtrs(size, GPUMemory, []unsafe.Pointer{
		base,
		unsafe.Pointer(uintptr(base) + stride),
		unsafe.Pointer(uintptr(base) + 2*stride),
	})
	dst := NewSlice(3, size)
	Copy(dst, src)
	if dtoHCalls != 1 {
		t.Fatalf("contiguous vector DtoH used %d transfers, want 1", dtoHCalls)
	}
	for c := range dst.Host() {
		for i, got := range dst.Host()[c] {
			want := backing[c*length+i]
			if got != want {
				t.Fatalf("component %d element %d = %v, want %v", c, i, got, want)
			}
		}
	}
	src.Disable() // fake GPU backing is owned by this test, not the callback
}

func TestGPUComponentViewsNeverFreePointers(t *testing.T) {
	oldFree, oldFreeHost := memFree, memFreeHost
	oldCopy, oldDtoH, oldHtoD := memCpy, memCpyDtoH, memCpyHtoD
	defer func() {
		memFree, memFreeHost = oldFree, oldFreeHost
		memCpy, memCpyDtoH, memCpyHtoD = oldCopy, oldDtoH, oldHtoD
	}()

	var freed []unsafe.Pointer
	copyBytes := func(dst, src unsafe.Pointer, bytes int64) {
		copy(unsafe.Slice((*byte)(dst), int(bytes)), unsafe.Slice((*byte)(src), int(bytes)))
	}
	EnableGPU(
		func(ptr unsafe.Pointer) { freed = append(freed, ptr) },
		func(unsafe.Pointer) {},
		copyBytes,
		copyBytes,
		copyBytes,
	)

	mesh := [3]int{4, 1, 1}
	backing := []float32{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23}
	base := unsafe.Pointer(&backing[0])
	ptrs := []unsafe.Pointer{
		base,
		unsafe.Pointer(&backing[4]),
		unsafe.Pointer(&backing[8]),
	}
	owner := SliceFromContiguousPtrs(mesh, GPUMemory, ptrs)
	if !owner.OwnsStorage() || !owner.Contiguous() {
		t.Fatal("contiguous GPU slice should own one allocation")
	}

	first := owner.Comp(0)
	interior := owner.Comp(1)
	if first.OwnsStorage() || interior.OwnsStorage() {
		t.Fatal("component views unexpectedly own GPU storage")
	}

	host := NewSlice(1, mesh)
	Copy(host, interior)
	for i, want := range []float32{10, 11, 12, 13} {
		if got := host.Host()[0][i]; got != want {
			t.Fatalf("copied component[%d] = %v, want %v", i, got, want)
		}
	}
	host.Free()

	first.Free()    // even the base-address component is a non-owning view
	interior.Free() // and an interior component must not reach the allocator
	if len(freed) != 0 {
		t.Fatalf("freeing component views released %d GPU pointers", len(freed))
	}

	lateView := owner.Comp(2)
	owner.Free()
	owner.Free()
	if len(freed) != 1 || freed[0] != base {
		t.Fatalf("contiguous owner freed %v, want base pointer %v exactly once", freed, base)
	}
	lateView.Free() // a stale header must not attempt a second/interior free
	lateView.Free()
	if len(freed) != 1 {
		t.Fatalf("freeing view after parent caused a double free: %d calls", len(freed))
	}
	runtime.KeepAlive(backing)
}

func TestNonContiguousGPUOwnerFreesEachAllocationButViewFreesNone(t *testing.T) {
	oldFree := memFree
	defer func() { memFree = oldFree }()

	var freed []unsafe.Pointer
	memFree = func(ptr unsafe.Pointer) { freed = append(freed, ptr) }
	a, b := new(float32), new(float32)
	owner := SliceFromPtrs([3]int{1, 1, 1}, GPUMemory, []unsafe.Pointer{unsafe.Pointer(a), unsafe.Pointer(b)})
	view := owner.Comp(1)
	view.Free()
	if len(freed) != 0 {
		t.Fatal("non-contiguous component view reached allocator")
	}
	owner.Free()
	if len(freed) != 2 || freed[0] != unsafe.Pointer(a) || freed[1] != unsafe.Pointer(b) {
		t.Fatalf("non-contiguous owner freed %v, want both component bases", freed)
	}
}
