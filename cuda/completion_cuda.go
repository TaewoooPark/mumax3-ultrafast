//go:build !darwin || !arm64 || !cgo

package cuda

// CUDA's synchronous host-to-device upload is ordered against the legacy
// default stream, so an explicit retirement token is unnecessary there.
type Completion struct{}

func RecordCompletion() Completion { return Completion{} }
func (Completion) Valid() bool     { return false }
func (Completion) Ready() bool     { return true }
func (Completion) Wait()           {}
func (*Completion) Free()          {}
