package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"reflect"
)

// Arbitrary physical quantity.
type Quantity interface {
	NComp() int
	EvalTo(dst *data.Slice)
}

func MeshSize() [3]int {
	return Mesh().Size()
}

func SizeOf(q Quantity) [3]int {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Mesh() *data.Mesh
	}); ok {
		return s.Mesh().Size()
	}
	// otherwise: default mesh
	return MeshSize()
}

func AverageOf(q Quantity) []float64 {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		average() []float64
	}); ok {
		return s.average()
	}
	// otherwise: default mesh
	buf := ValueOf(q)
	defer cuda.Recycle(buf)
	return sAverageMagnet(buf)
}

func NameOf(q Quantity) string {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Name() string
	}); ok {
		return s.Name()
	}
	return "unnamed." + reflect.TypeOf(q).String()
}

func UnitOf(q Quantity) string {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Unit() string
	}); ok {
		return s.Unit()
	}
	return "?"
}

func MeshOf(q Quantity) *data.Mesh {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Mesh() *data.Mesh
	}); ok {
		return s.Mesh()
	}
	return Mesh()
}

func ValueOf(q Quantity) *data.Slice {
	// TODO: check for Buffered() implementation
	buf := cuda.Buffer(q.NComp(), SizeOf(q))
	q.EvalTo(buf)
	return buf
}

type sliceProvider interface {
	Slice() (*data.Slice, bool)
}

// HostCopyOf snapshots a quantity without an unnecessary GPU-to-GPU staging
// copy when the quantity already exposes its backing slice. This is especially
// important for Save(M), where magnetization is stable until HostCopy returns
// and the host snapshot itself provides the asynchronous-I/O isolation.
func HostCopyOf(q Quantity) *data.Slice {
	if s, recycle, ok := quantitySlice(q); ok {
		host := s.HostCopy()
		if recycle {
			cuda.Recycle(s)
		}
		return host
	}
	buf := ValueOf(q)
	host := buf.HostCopy()
	cuda.Recycle(buf)
	return host
}

func quantitySlice(q Quantity) (*data.Slice, bool, bool) {
	if provider, ok := q.(sliceProvider); ok {
		s, recycle := provider.Slice()
		return s, recycle, true
	}
	// VectorField and ScalarField deliberately expose only Quantity's method
	// set. Unwrap them so fieldFunc.Slice and other optimized providers remain
	// visible to the output path.
	switch field := q.(type) {
	case VectorField:
		return quantitySlice(field.Quantity)
	case *VectorField:
		return quantitySlice(field.Quantity)
	case ScalarField:
		return quantitySlice(field.Quantity)
	case *ScalarField:
		return quantitySlice(field.Quantity)
	default:
		return nil, false, false
	}
}

// Temporary shim to fit Slice into EvalTo
func EvalTo(q interface {
	Slice() (*data.Slice, bool)
}, dst *data.Slice) {
	v, r := q.Slice()
	if r {
		defer cuda.Recycle(v)
	}
	data.Copy(dst, v)
}
