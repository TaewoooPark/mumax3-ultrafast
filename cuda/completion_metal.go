//go:build darwin && arm64 && cgo

package cuda

import (
	"github.com/mumax/3/cuda/metal"
	"github.com/mumax/3/util"
)

// Completion is a non-submitting snapshot of Metal's ordered queue tail.
// Slots backed by shared memory use it to prove that earlier GPU readers have
// stopped before the CPU overwrites the storage.
type Completion struct {
	token metal.Completion
}

func RecordCompletion() Completion {
	token, err := metal.RecordCompletion()
	util.PanicErr(err)
	return Completion{token: token}
}

func (completion Completion) Valid() bool {
	return completion.token.Valid()
}

func (completion Completion) Ready() bool {
	if !completion.token.Valid() {
		return true
	}
	ready, err := completion.token.Ready()
	util.PanicErr(err)
	return ready
}

func (completion Completion) Wait() {
	if completion.token.Valid() {
		util.PanicErr(completion.token.Wait())
	}
}

func (completion *Completion) Free() {
	if completion == nil || !completion.token.Valid() {
		return
	}
	completion.token.Close()
	completion.token = metal.Completion{}
}
