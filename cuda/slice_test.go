package cuda

import (
	"testing"

	"github.com/mumax/3/data"
)

func TestSlice(t *testing.T) {
	N0, N1, N2 := 2, 4, 8
	m := [3]int{N0, N1, N2}
	N := N0 * N1 * N2

	a := NewSlice(3, m)
	defer a.Free()
	Memset(a, 1, 2, 3)

	if a.GPUAccess() == false {
		t.Fail()
	}
	if a.Len() != N {
		t.Fail()
	}
	if a.NComp() != 3 {
		t.Fail()
	}

	b := a.Comp(1)
	if !a.OwnsStorage() {
		t.Error("owning slice does not report storage ownership")
	}
	if b.OwnsStorage() {
		t.Error("component view unexpectedly reports storage ownership")
	}
	if b.GPUAccess() == false {
		t.Error("b.GPUAccess", b.GPUAccess())
	}
	if b.Len() != N {
		t.Error("b.Len", b.Len())
	}
	if b.NComp() != 1 {
		t.Error("b.NComp", b.NComp())
	}
	if b.Size() != a.Size() {
		t.Fail()
	}

	Memset(b, 9)
	b.Free()
	b.Free()
	if b.NComp() != 0 {
		t.Error("freed component view is still enabled")
	}
	if a.NComp() != 3 || !a.OwnsStorage() {
		t.Fatal("freeing component view disabled the owning GPU slice")
	}
	host := a.HostCopy()
	defer host.Free()
	if got := host.Host()[0][0]; got != 1 {
		t.Errorf("component 0 changed after view free: got %v", got)
	}
	if got := host.Host()[1][0]; got != 9 {
		t.Errorf("component view write was lost: got %v", got)
	}
	if got := host.Host()[2][0]; got != 3 {
		t.Errorf("component 2 changed after view free: got %v", got)
	}
}

func TestCpy(t *testing.T) {
	N0, N1, N2 := 2, 4, 32
	N := N0 * N1 * N2
	mesh := [3]int{N0, N1, N2}

	h1 := make([]float32, N)
	for i := range h1 {
		h1[i] = float32(i)
	}
	hs := sliceFromList([][]float32{h1}, mesh)

	d := NewSlice(1, mesh)
	data.Copy(d, hs)

	d2 := NewSlice(1, mesh)
	data.Copy(d2, d)

	h2 := data.NewSlice(1, mesh)
	data.Copy(h2, d2)

	res := h2.Host()[0]
	for i := range res {
		if res[i] != h1[i] {
			t.Fail()
		}
	}
}

func TestSliceFree(t *testing.T) {
	N0, N1, N2 := 128, 1024, 1024
	m := [3]int{N0, N1, N2}
	N := 17
	// not freeing would attempt to allocate 17GB.
	for i := 0; i < N; i++ {
		a := NewSlice(2, m)
		a.Free()
	}
	a := NewSlice(2, m)
	a.Free()
	a.Free() // test double-free
}

func TestSliceHost(t *testing.T) {
	N0, N1, N2 := 1, 10, 10
	m := [3]int{N0, N1, N2}
	a := NewSlice(3, m)
	defer a.Free()

	b := a.HostCopy().Host()
	if b[0][0] != 0 || b[1][42] != 0 || b[2][99] != 0 {
		t.Error("slice not inited to zero")
	}

	Memset(a, 1, 2, 3)
	b = a.HostCopy().Host()
	if b[0][0] != 1 || b[1][42] != 2 || b[2][99] != 3 {
		t.Error("slice memset")
	}
}

func TestContiguousZeroMulAndDiv(t *testing.T) {
	size := [3]int{1, 1, 37}
	a := NewSlice(3, size)
	b := NewSlice(3, size)
	dst := NewSlice(3, size)
	defer a.Free()
	defer b.Free()
	defer dst.Free()

	Memset(a, 2, 4, 6)
	Memset(b, 3, 5, 7)
	Mul(dst, a, b)
	host := dst.HostCopy()
	for c, want := range []float32{6, 20, 42} {
		for i, got := range host.Host()[c] {
			if got != want {
				t.Fatalf("Mul component %d element %d = %v, want %v", c, i, got, want)
			}
		}
	}
	host.Free()

	Div(dst, b, a)
	host = dst.HostCopy()
	for c, want := range []float32{1.5, 1.25, 7.0 / 6.0} {
		for i, got := range host.Host()[c] {
			if got != want {
				t.Fatalf("Div component %d element %d = %v, want %v", c, i, got, want)
			}
		}
	}
	host.Free()

	Zero(dst)
	host = dst.HostCopy()
	defer host.Free()
	for c := range host.Host() {
		for i, got := range host.Host()[c] {
			if got != 0 {
				t.Fatalf("Zero component %d element %d = %v", c, i, got)
			}
		}
	}
}
