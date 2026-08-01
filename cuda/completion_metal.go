//go:build darwin && arm64 && cgo

package cuda

import (
	"unsafe"

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

// CloseQueueBatch commits everything encoded so far without waiting for it,
// which ends the current command buffer. A Completion recorded just before this
// therefore covers exactly the work encoded up to that point, and anything
// encoded afterwards lands in a fresh command buffer the token does not include
// - which is what makes waiting the token overlap with later work instead of
// waiting for it too.
func CloseQueueBatch() {
	util.PanicErr(metal.Flush())
}

// ReadHostAfter copies from a device allocation without draining the queue,
// having first waited for the recorded work that writes it. Allocations are
// shared memory, so once that work is done the bytes are simply there.
func ReadHostAfter(dst, src unsafe.Pointer, bytes int64, completion *Completion) {
	completion.Wait()
	util.PanicErr(metal.CopyToHostUnordered(dst, src, bytes))
}

// TargetedReadbackSupported reports whether a reduction can be read by waiting
// only its own command buffer.
const TargetedReadbackSupported = true
