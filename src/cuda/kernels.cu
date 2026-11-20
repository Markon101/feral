// FERAL resident FP32 CUDA kernels. Batched, frame-major, no fast-math intrinsics.
// These mirror src/gpu/kernels.cl operation-for-operation so the CPU reference remains
// the numerical oracle. Tiles are 16x16 (256 threads); shared memory is 2 KiB.
#include <cuda_runtime.h>
#include "conv32.cuh"

#define TILE 16

// Conditional flow input construction. `data` is [clean | noise | mask | trig] where each
// waveform block is contiguous per item and trig holds t, 1-t, sin, cos per item.
extern "C" __global__ void feral_features(
    const float *__restrict__ data, float *__restrict__ x,
    unsigned n, unsigned block, int pre, float data_scale, float prior_std,
    unsigned frames, unsigned total_frames) {
  unsigned channels = 3u * block + 4u;
  unsigned c = blockIdx.y * blockDim.x + threadIdx.x;
  unsigned p = blockIdx.x * blockDim.y + threadIdx.y;
  if (c >= channels || p >= total_frames) return;
  unsigned id = p * channels + c;
  unsigned item = p / frames;
  if (c < block) {
    unsigned i = p * block + c;
    float y = data[i], z = data[n + i], m = data[2u * n + i];
    float t = data[3u * n + 4u * item];
    float xt = (m == 1.0f) ? y : (1.0f - t) * z + t * y;
    if (pre) {
      float omt = 1.0f - t;
      float a = data_scale * data_scale, b = prior_std * prior_std;
      float cin = 1.0f / sqrtf(t * t * a + omt * omt * b);
      x[id] = (m == 1.0f) ? y / data_scale : xt * cin;
    } else {
      x[id] = xt;
    }
  } else if (c < 2u * block) {
    unsigned i = p * block + (c - block);
    float y = data[i];
    x[id] = (pre ? y / data_scale : y) * data[2u * n + i];
  } else if (c < 3u * block) {
    x[id] = data[2u * n + p * block + (c - 2u * block)];
  } else {
    x[id] = data[3u * n + 4u * item + (c - 3u * block)];
  }
}

// Average pooling per item; the final group uses its actual length, never padded zeros.
extern "C" __global__ void feral_pool(const float *__restrict__ x, float *__restrict__ y,
                                      unsigned frames, unsigned channels, unsigned scale,
                                      unsigned items) {
  unsigned groups = (frames + scale - 1u) / scale;
  unsigned per = groups * channels;
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= per * items) return;
  unsigned item = id / per, r0 = id % per, q = r0 / channels, c = r0 % channels;
  unsigned count = min(scale, frames - q * scale);
  float sum = 0.f;
  for (unsigned r = 0; r < count; r++) {
    sum += x[(item * frames + q * scale + r) * channels + c] / (float)count;
  }
  y[id] = sum;
}

// Unknown-sample MSE and its derivative. `unknown` holds the per-item unknown count so the
// reported loss is the mean over batch items; weight gradients divide by batch as well.
// Residual at one sample, in whichever space the parameterization trains in.
__device__ __forceinline__ void feral_precond_coeffs(
    float t, float data_scale, float prior_std, float *cskip, float *cout) {
  float omt = 1.0f - t;
  float a = data_scale * data_scale, b = prior_std * prior_std;
  float d = t * t * a + omt * omt * b;
  *cskip = (t * a - omt * b) / d;
  *cout = data_scale * prior_std / sqrtf(d);
}

__device__ __forceinline__ float feral_residual(const float *__restrict__ prediction,
                                                const float *__restrict__ data, unsigned total,
                                                unsigned j, int pre, float t, float cskip,
                                                float cout) {
  float y = data[j], z = data[total + j], m = data[2u * total + j];
  float xt = (m == 1.0f) ? y : (1.0f - t) * z + t * y;
  float target = pre ? (y - z - cskip * xt) / cout : (y - z);
  return prediction[j] - target;
}

extern "C" __global__ void feral_loss_grad(const float *__restrict__ prediction,
                                           const float *__restrict__ data,
                                           const float *__restrict__ unknown,
                                           float *__restrict__ dy, float *__restrict__ partial,
                                           unsigned n, int pre, float data_scale,
                                           float prior_std, float detail, unsigned items) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x;
  __shared__ float scratch[256];
  float value = 0.f;
  unsigned total = items * n;
  if (id < total) {
    unsigned item = id / n;
    float y = data[id], z = data[total + id], m = data[2u * total + id];
    float t = data[3u * total + 4u * item];
    float u = fmaxf(unknown[item], 1.0f);
    float cskip = 0.0f, cout = 1.0f;
    if (pre) feral_precond_coeffs(t, data_scale, prior_std, &cskip, &cout);
    float xt = (m == 1.0f) ? y : (1.0f - t) * z + t * y;
    float target = pre ? (y - z - cskip * xt) / cout : (y - z);
    float r = prediction[id] - target;
    float d = r * (1.0f - m);
    float dyv = 2.0f * d / u;
    value = d * d / u;
    // Detail term: mean squared first difference of the residual over pairs where both samples
    // are unknown, so known regions never leak in. Mirrors flow::loss_gradient_detailed.
    if (detail > 0.f) {
      unsigned base = item * n, end = base + n;
      if (m == 0.f && id + 1 < end && data[2u * total + id + 1] == 0.f) {
        float delta = feral_residual(prediction, data, total, id + 1, pre, t, cskip, cout) - r;
        value += detail * delta * delta / u;
        dyv -= 2.f * detail * delta / u;
      }
      if (m == 0.f && id > base && data[2u * total + id - 1] == 0.f) {
        float delta = r - feral_residual(prediction, data, total, id - 1, pre, t, cskip, cout);
        dyv += 2.f * detail * delta / u;
      }
    }
    dy[id] = dyv;
  }
  scratch[lane] = value;
  __syncthreads();
  for (unsigned step = blockDim.x / 2; step > 0; step >>= 1) {
    if (lane < step) scratch[lane] += scratch[lane + step];
    __syncthreads();
  }
  if (lane == 0) partial[blockIdx.x] = scratch[0];
}

// tanh derivative, reverse of pooled upsampling, and the residual 0.5 factor.
extern "C" __global__ void feral_activation_grad(const float *__restrict__ dh,
                                                 const float *__restrict__ z,
                                                 float *__restrict__ dz, unsigned frames,
                                                 unsigned channels, unsigned scale, int residual,
                                                 unsigned items) {
  unsigned cf = (frames + scale - 1u) / scale, per = cf * channels;
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= per * items) return;
  unsigned item = id / per, r0 = id % per, q = r0 / channels, c = r0 % channels;
  unsigned count = min(scale, frames - q * scale);
  float g = 0.f;
  for (unsigned r = 0; r < count; r++) {
    g += dh[(item * frames + q * scale + r) * channels + c];
  }
  float a = z[id];
  dz[id] = g * (1.0f - a * a) * (residual ? 0.5f : 1.0f);
}

// One block: gradient norm, clipped scale, mean loss over batch and a nonfinite flag.
extern "C" __global__ void feral_statistics(const float *__restrict__ grad,
                                            const float *__restrict__ loss_parts,
                                            float *__restrict__ stats, unsigned parameters,
                                            unsigned parts, float inv_batch) {
  unsigned lane = threadIdx.x;
  __shared__ float sums[256], losses[256];
  __shared__ int bad[256];
  float sum = 0.f, loss = 0.f;
  int invalid = 0;
  for (unsigned i = lane; i < parameters; i += 256) {
    float g = grad[i];
    sum += g * g;
    invalid |= !isfinite(g);
  }
  for (unsigned i = lane; i < parts; i += 256) loss += loss_parts[i];
  sums[lane] = sum;
  losses[lane] = loss;
  bad[lane] = invalid;
  __syncthreads();
  for (unsigned s = 128; s > 0; s >>= 1) {
    if (lane < s) {
      sums[lane] += sums[lane + s];
      losses[lane] += losses[lane + s];
      bad[lane] |= bad[lane + s];
    }
    __syncthreads();
  }
  if (lane == 0) {
    float l = losses[0] * inv_batch;
    stats[0] = l;
    stats[1] = sqrtf(sums[0]);
    stats[2] = 1.0f / fmaxf(1.0f, stats[1]);
    stats[3] = 0.0f;
    if (bad[0] || !isfinite(l) || !isfinite(stats[1])) stats[3] = 1.0f;
  }
}

// stats[3] is the atomic error flag; never read it nonatomically here.
extern "C" __global__ void feral_adam(float *__restrict__ weights, float *__restrict__ m,
                                      float *__restrict__ v, const float *__restrict__ grad,
                                      float *__restrict__ stats, unsigned n, float lr, float c1,
                                      float c2) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !isfinite(stats[0]) || !isfinite(stats[1])) return;
  float g = grad[i] * stats[2], mm = 0.9f * m[i] + 0.1f * g,
        vv = 0.999f * v[i] + 0.001f * g * g;
  float p = weights[i] - lr * (mm / c1) / (sqrtf(vv / c2) + 1e-8f);
  m[i] = mm;
  v[i] = vv;
  weights[i] = p;
  if (!isfinite(p) || !isfinite(mm) || !isfinite(vv)) atomicOr((int *)&stats[3], 1);
}
