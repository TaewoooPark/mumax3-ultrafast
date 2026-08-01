#include <stdint.h>
#include "float3.h"

// Steepest descent energy minimizer taking its Barzilai-Borwein step size from
// device memory instead of from a host scalar.
//
// nomPartials and divPartials are the un-combined partial slots of the two dot
// products the step size is built from. Every thread re-adds the same few slots
// in the same index order the host would, so the scalar is bit-identical to
// computing it on the CPU - and the host never has to read the reductions back,
// which is what otherwise forces a full pipeline drain on every iteration.
//
// The redundant re-add is deliberate. Reducing the slots in one place would need
// either a second dispatch or a grid-wide barrier, whereas each thread summing
// two dozen floats costs nothing against the nine buffer accesses below.
//
// Body identical to minimize() apart from where dt comes from:
//   m = 1 / (4 + t^2(m x H)^2) [{4 - t^2(m x H)^2} m - 4t(m x m x H)]
// note: torque from LLNoPrecess has negative sign
extern "C" __global__ void
bbminimize(float* __restrict__ mx,  float* __restrict__  my,  float* __restrict__ mz,
           float* __restrict__ m0x, float* __restrict__  m0y, float* __restrict__ m0z,
           float* __restrict__ tx,  float* __restrict__  ty,  float* __restrict__ tz,
           float* __restrict__ nomPartials, float* __restrict__ divPartials,
           int slots, int N) {

    float nom = 0.0f;
    float divisorSum = 0.0f;
    for (int s = 0; s < slots; s++) {
        nom += nomPartials[s];
        divisorSum += divPartials[s];
    }
    // Matches the host fallback, including its division-by-zero guard.
    float dt = (divisorSum != 0.0f) ? (nom / divisorSum) : 1e-4f;

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 t = {tx[i], ty[i], tz[i]};

        float t2 = dt*dt*dot(t, t);
        float3 result = (4 - t2) * m0 + 4 * dt * t;
        float divisor = 4 + t2;

        mx[i] = result.x / divisor;
        my[i] = result.y / divisor;
        mz[i] = result.z / divisor;
    }
}
