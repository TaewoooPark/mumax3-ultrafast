//go:build darwin && arm64 && cgo && go1.24

package metal

/*
// Go 1.24 added these cgo escape/callback declarations. mr_launch_handle
// consumes the argument table and error-result address synchronously: Metal
// copies scalar bytes during encoding, buffer arguments are resolved to native
// MTLBuffer objects, and neither Go address is retained after the call. The
// Objective-C++ implementation also has no route back into Go.
//
// Keeping the directives in a release-tagged file preserves the module's Go
// 1.22.4 minimum; older cgo parsers exclude this file before reading them.
#cgo noescape mr_launch_handle
#cgo nocallback mr_launch_handle
#include "metal_runtime.h"
*/
import "C"
