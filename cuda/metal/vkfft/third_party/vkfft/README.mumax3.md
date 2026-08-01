# Vendored VkFFT sources

This directory vendors the unmodified VkFFT 1.3.4 sources and its bundled
metal-cpp dependency from upstream commit
`066a17c17068c0f11c9298d848c2976c71fad1c1`.

- VkFFT is Copyright (c) 2020-present Dmitrii Tolmachev and licensed under the
  MIT License in `LICENSE.VkFFT`.
- metal-cpp is Copyright (c) 2020 Apple Inc. and licensed under the Apache
  License 2.0 in `metal-cpp/LICENSE.txt`.

The MuMax3 bridge includes these headers without modifying them. Updating the
dependency should replace the complete `vkFFT` and `metal-cpp` directories,
update the pinned commit above, and rerun the Metal FFT correctness and
end-to-end demagnetization benchmarks.
