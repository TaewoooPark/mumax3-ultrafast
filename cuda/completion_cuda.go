//go:build !darwin || !arm64 || !cgo

package cuda

import "unsafe"

// CUDA's synchronous host-to-device upload is ordered against the legacy
// default stream, so an explicit retirement token is unnecessary there.
type Completion struct{}

func RecordCompletion() Completion { return Completion{} }
func (Completion) Valid() bool     { return false }
func (Completion) Ready() bool     { return true }
func (Completion) Wait()           {}
func (*Completion) Free()          {}

// CloseQueueBatch has no counterpart on the legacy default stream, where work is
// already submitted as it is issued.
func CloseQueueBatch() {}

// ReadHostAfter falls back to the ordered copy, which synchronizes the stream.
func ReadHostAfter(dst, src unsafe.Pointer, bytes int64, completion *Completion) {
	MemCpyDtoH(dst, src, bytes)
}

// TargetedReadbackSupported reports whether a reduction can be read by waiting
// only its own command buffer. The default stream offers no such scope.
const TargetedReadbackSupported = false
