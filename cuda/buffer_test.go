package cuda

import (
	"testing"

	"github.com/mumax/3/data"
)

// In case of memory leak, this will crash
func TestBuffer(t *testing.T) {
	m1 := [3]int{2, 1024, 2048}
	m2 := [3]int{4, 1024, 2048}
	a := Buffer(3, m1)
	b := Buffer(3, m2)
	c := Buffer(1, m1)
	d := Buffer(2, m2)

	Recycle(a)
	Recycle(b)
	Recycle(c)
	Recycle(d)

	for i := 0; i < 10000; i++ {
		b := Buffer(3, m2)
		Recycle(b)
	}
}

func BenchmarkBuffer(b *testing.B) {
	b.StopTimer()
	m := [3]int{2, 1024, 2048}
	a := Buffer(3, m)
	Recycle(a)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		a := Buffer(3, m)
		Recycle(a)
	}
}

func TestBufferComponentViewCannotBeRecycled(t *testing.T) {
	size := [3]int{7, 5, 1}
	owner := Buffer(3, size)
	base := owner.DevPtr(0)

	view := owner.Comp(0) // base address is still non-owning
	mustPanic(t, func() { Recycle(view) })
	if view.NComp() != 1 {
		t.Fatal("failed Recycle unexpectedly disabled the view")
	}
	view.Free()
	Recycle(view) // a disabled view is the same harmless no-op as any freed Slice

	interior := owner.Comp(2)
	mustPanic(t, func() { Recycle(interior) })
	interior.Free()

	Memset(owner, 4, 5, 6)
	host := data.NewSlice(3, size)
	data.Copy(host, owner)
	defer host.Free()
	if got := host.Host()[0][0]; got != 4 {
		t.Fatalf("owner component 0 = %v after view operations, want 4", got)
	}
	if got := host.Host()[2][0]; got != 6 {
		t.Fatalf("owner component 2 = %v after view operations, want 6", got)
	}

	Recycle(owner)
	Recycle(owner) // owner recycle is idempotent after it disables the header
	reused := Buffer(3, size)
	if got := reused.DevPtr(0); got != base {
		t.Fatalf("pool did not reuse complete allocation: got %v, want %v", got, base)
	}
	Recycle(reused)
}

func TestPooledOwnerMayBeFreedExactlyOnce(t *testing.T) {
	owner := Buffer(3, [3]int{11, 3, 1})
	owner.Free()
	owner.Free()
	if owner.NComp() != 0 || owner.OwnsStorage() {
		t.Fatal("double-freed pooled owner was not disabled")
	}
}

func mustPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("call did not panic")
		}
	}()
	f()
}
