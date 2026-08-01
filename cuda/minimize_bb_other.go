//go:build !darwin || !arm64 || !cgo

package cuda

import "github.com/mumax/3/data"

// BBStepSupported reports whether the Barzilai-Borwein step size can stay on the
// device. The bbminimize launch wrapper is produced by cuda/Makefile, which needs
// nvcc, and only the Metal wrapper is checked in - so on CUDA the minimizer keeps
// computing the step size on the host. Regenerating the CUDA wrappers is enough
// to enable this path here as well.
const BBStepSupported = false

// BBStep is a placeholder so callers compile without build tags of their own.
type BBStep struct{}

func NewBBStep() *BBStep { panic("cuda: device-resident BB step is unavailable on this backend") }

func (step *BBStep) Seed(h float32) {}

func (step *BBStep) Update(dm, dk *data.Slice, longFormula bool) {}

func (step *BBStep) Free() {}

func MinimizeBB(m, m0, torque *data.Slice, step *BBStep) {
	panic("cuda: device-resident BB step is unavailable on this backend")
}
