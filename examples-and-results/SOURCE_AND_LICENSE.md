# Official example source, attribution, and license

## Example input scripts

The files `example1.mx3` through `example15.mx3` in
[`official-examples/`](official-examples/) are reproduced verbatim from the
[official mumax³ examples page](https://mumax.github.io/examples.html).
The page is generated from
[`doc/templates/examples-template.html`](https://github.com/mumax/3/blob/f656494b29516bead825b444b1f0b38c6e6c7dbf/doc/templates/examples-template.html)
in the upstream `mumax/3` repository. A byte-for-byte comparison against its
15 template blocks passed for all 15 checked-in inputs.

That upstream source is licensed under **GNU GPL version 3 or any later
version (`GPL-3.0-or-later`)**, including the additional permission for linking
with NVIDIA CUDA libraries stated by the upstream licensors. The complete GPLv3
text is in this repository's [`LICENSE`](../LICENSE), while the preserved
upstream notice and CUDA permission are in [`NOTICE`](../NOTICE). The original
notice is also available in the
[upstream LICENSE](https://github.com/mumax/3/blob/f656494b29516bead825b444b1f0b38c6e6c7dbf/LICENSE).

Upstream attribution, as stated in that license:

> Mumax3 GPU-accelerated micromagnetic simulator
>
> Copyright (C) 2012-2014 Arne Vansteenkiste.
> Contributions by Ahmad Syukri, Colin Jermain, Jonathan Leliaert,
> Mykola Dvornik.

The scripts remain unmodified so that they are exact test inputs; SPDX comments
were deliberately not inserted into the files themselves. This notice supplies
the license and attribution for the whole set.
The archived `official-examples.html` snapshot is generated from the same
GPL-covered upstream documentation and is included under the same notice.

## Supporting fixtures and generated results

`mask.png` and `myfile.ovf` are copies of
`test/testdata/mask.png` and `test/testdata/randommag4x4x1.ovf` from this
GPL-licensed source tree. They are not presented as files downloaded from the
examples website.

The `output.out/` directories, `run.log` files, `PASS` markers, and JSON
summaries are locally generated reproducibility artifacts from the Apple M4
Metal run. They do not alter or replace the license of the upstream input
scripts, and their inclusion does not imply endorsement by the upstream mumax³
authors.
