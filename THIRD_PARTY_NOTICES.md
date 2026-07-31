# Third-party notices

This file records third-party material used by mumax³ for macOS. It applies to
the source distribution and must accompany prebuilt binary distributions.
The license for the combined mumax³ work is in [LICENSE](LICENSE), and the
upstream notice plus the macOS modification notice are in [NOTICE](NOTICE).

## Random123 Philox

The Metal thermal-noise implementation contains a language-level translation of
the Philox4x32-10 round function and constants from
[Random123](https://github.com/DEShawResearch/random123). Its known-answer
tests use the Philox4x32-10 zero-input vector published in Random123's
`tests/kat_vectors`.

Affected source files:

- `cuda/metal/rng/philox.go`
- `cuda/metal/rng/bridge_darwin_arm64.cc`
- `cuda/metal/rng/philox_test.go`
- `cuda/metal/rng/gpu_darwin_test.go`

Random123 is distributed under the BSD 3-Clause license reproduced verbatim
below.

```text
/** @page LICENSE
Copyright 2010-2012, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
```

## The Go programming language

Prebuilt mumax³ binaries contain portions of the Go runtime and standard
library. Source-only installation does not redistribute the Go toolchain, but
every prebuilt binary package must include this notice.

Go is distributed under the BSD 3-Clause license reproduced verbatim below.
The upstream project also publishes a
[patent grant](https://go.dev/PATENTS).

```text
Copyright 2009 The Go Authors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * Neither the name of Google LLC nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## cuda5 Go bindings

The legacy CUDA-only driver, cuFFT, and cuRAND Go bindings retained under
`cuda/cu/`, `cuda/cufft/`, and `cuda/curand/` originated in
[barnex/cuda5](https://github.com/barnex/cuda5). Their retained source notice
identifies the FreeBSD license, commonly designated BSD-2-Clause. Because the
referenced `LICENSE.txt` is absent from both repositories, the standard
BSD-2-Clause/FreeBSD text matching that explicit license designation is
reproduced below. This does not alter the original terms.

The Apple Silicon `*_metal.go`, `metal_compat.go`, and related new files were
not part of cuda5; they remain covered by the combined project's
GPL-3.0-or-later terms.

```text
Copyright 2011 Arne Vansteenkiste (barnex@gmail.com). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## SVGo

The vendored `svgo/` package is SVGo by Anthony Starks:

- Source: [github.com/ajstarks/svgo](https://github.com/ajstarks/svgo)
- License: [Creative Commons Attribution 3.0 United States](https://creativecommons.org/licenses/by/3.0/us/)
- Local notice: [svgo/LICENSE](svgo/LICENSE)

The package name, author, source, and license URI above provide the attribution
required for its use in this distribution.

## Freetype-Go

The vendored `freetype/` raster package is Freetype-Go, copyright Google Inc.,
Jeff R. Allen, Rémy Oudompheng, Roger Peppe, and the Freetype-Go authors.

Freetype-Go offers a choice of the FreeType License or GNU GPL version 2 or any
later version. This combined GPLv3-or-later distribution uses the
GPL-2.0-or-later option. Its original dual-license notice and author records
remain available in:

- [freetype/LICENSE](freetype/LICENSE)
- [freetype/AUTHORS](freetype/AUTHORS)
- [freetype/CONTRIBUTORS](freetype/CONTRIBUTORS)
