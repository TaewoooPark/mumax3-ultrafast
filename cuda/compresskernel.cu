#include "stencil.h"

// Extract the non-redundant real half of a Hermitian demag-kernel spectrum.
// src is complex data stored as interleaved floats. The Y and Z dimensions of
// dst cover only the leading mirror-symmetric half. imag receives the
// normalized imaginary magnitude used by the initialization sanity check.
extern "C" __global__ void
compresskernel(float* __restrict__ dst,
               float* __restrict__ imag,
               int Dx, int Dy, int Dz,
               float* __restrict__ src,
               int Sx, int Sy, int Sz,
               float realScale, float imagScale) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < Dx && iy < Dy && iz < Dz) {
        int d = index(ix, iy, iz, Dx, Dy, Dz);
        int s = index(2 * ix, iy, iz, Sx, Sy, Sz);
        dst[d] = src[s] * realScale;
        imag[d] = src[s + 1] * imagScale;
    }
}
