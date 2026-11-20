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

// Add host-computed spectral endpoint gradient into dy and fold the batch spectral loss sum into
// loss_parts[0] (which feral_statistics sums and divides by items).
extern "C" __global__ void feral_add_spectral(float *__restrict__ dy,
                                              const float *__restrict__ dy_spec,
                                              float *__restrict__ loss_parts, float spec_loss_sum,
                                              unsigned total) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total) dy[id] += dy_spec[id];
  if (id == 0u) loss_parts[0] += spec_loss_sum;
}

// Exponential moving average of weights: ema = decay * ema + (1 - decay) * weights.
extern "C" __global__ void feral_ema(float *__restrict__ ema, const float *__restrict__ weights,
                                     const float *__restrict__ stats, unsigned n, float decay) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !isfinite(stats[0]) || !isfinite(stats[1])) return;
  ema[i] = decay * ema[i] + (1.0f - decay) * weights[i];
}

// Sample-level residual conv helpers (sample_conv = K odd in 3..=15).
extern "C" __global__ void feral_unpack_stem_in(const float *__restrict__ x_features,
                                                float *__restrict__ stem_in, unsigned frames,
                                                unsigned block, unsigned items) {
  unsigned total = items * frames * block;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  unsigned p = idx / block;
  unsigned c = idx % block;
  unsigned channels = 3u * block + 4u;
  unsigned base = p * channels;
  stem_in[idx * 3u + 0u] = x_features[base + c];
  stem_in[idx * 3u + 1u] = x_features[base + block + c];
  stem_in[idx * 3u + 2u] = x_features[base + 2u * block + c];
}

extern "C" __global__ void feral_apply_stem_residual(float *__restrict__ x_features,
                                                     const float *__restrict__ stem_a1,
                                                     unsigned frames, unsigned block,
                                                     unsigned items) {
  unsigned total = items * frames * block;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  unsigned p = idx / block;
  unsigned c = idx % block;
  unsigned channels = 3u * block + 4u;
  x_features[p * channels + c] += 0.5f * stem_a1[idx];
}

extern "C" __global__ void feral_extract_stem_grad(const float *__restrict__ dh0,
                                                   const float *__restrict__ stem_a1,
                                                   float *__restrict__ dz_s1, unsigned frames,
                                                   unsigned block, unsigned items) {
  unsigned total = items * frames * block;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  unsigned p = idx / block;
  unsigned c = idx % block;
  unsigned channels = 3u * block + 4u;
  float a = stem_a1[idx];
  dz_s1[idx] = dh0[p * channels + c] * 0.5f * (1.0f - a * a);
}

extern "C" __global__ void feral_scale_half(const float *__restrict__ src,
                                            float *__restrict__ dst, unsigned total) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total) dst[id] = 0.5f * src[id];
}

extern "C" __global__ void feral_sample_conv_fwd(
    const float *__restrict__ x, const float *__restrict__ weights, float *__restrict__ y,
    unsigned n, unsigned ci, unsigned co, unsigned k, unsigned wo, unsigned bo, int act_tanh,
    const float *__restrict__ skip, float skip_scale, unsigned items) {
  unsigned total = items * n * co;
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= total) return;
  unsigned o = id % co;
  unsigned tmp = id / co;
  unsigned p = tmp % n;
  unsigned item = tmp / n;
  int half = (int)(k / 2u);
  float sum = weights[bo + o];
  for (unsigned kk = 0u; kk < k; kk++) {
    int q = (int)p + (int)kk - half;
    if (q < 0 || q >= (int)n) continue;
    const float *row = weights + wo + (o * k + kk) * ci;
    const float *xx = x + (item * n + (unsigned)q) * ci;
    for (unsigned i = 0u; i < ci; i++) sum += row[i] * xx[i];
  }
  if (act_tanh) sum = tanhf(sum);
  if (skip != nullptr) sum = skip[id] + skip_scale * sum;
  y[id] = sum;
}

extern "C" __global__ void feral_sample_conv_dx(
    const float *__restrict__ dz, const float *__restrict__ weights,
    const float *__restrict__ a_prev, const float *__restrict__ skip_grad, float *__restrict__ dx,
    unsigned n, unsigned ci, unsigned co, unsigned k, unsigned wo, int apply_tanh_deriv,
    unsigned items) {
  unsigned total = items * n * ci;
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= total) return;
  unsigned i = id % ci;
  unsigned tmp = id / ci;
  unsigned q = tmp % n;
  unsigned item = tmp / n;
  int half = (int)(k / 2u);
  float sum = 0.0f;
  for (unsigned kk = 0u; kk < k; kk++) {
    int p = (int)q + half - (int)kk;
    if (p < 0 || p >= (int)n) continue;
    const float *zz = dz + (item * n + (unsigned)p) * co;
    for (unsigned o = 0u; o < co; o++) {
      sum += zz[o] * weights[wo + (o * k + kk) * ci + i];
    }
  }
  if (apply_tanh_deriv) {
    float a = a_prev[id];
    sum *= (1.0f - a * a);
  }
  if (skip_grad != nullptr) sum += skip_grad[id];
  dx[id] = sum;
}

extern "C" __global__ void feral_sample_conv_dw(
    const float *__restrict__ x, const float *__restrict__ dz, float *__restrict__ grad,
    unsigned n, unsigned ci, unsigned co, unsigned k, unsigned wo, unsigned bo, float inv_batch,
    unsigned items) {
  unsigned bid = blockIdx.x;
  unsigned lane = threadIdx.x;
  unsigned weight_count = co * k * ci;
  __shared__ float scratch[256];
  float acc = 0.0f;
  unsigned total = items * n;
  if (bid < weight_count) {
    unsigned i = bid % ci;
    unsigned tmp = bid / ci;
    unsigned kk = tmp % k;
    unsigned o = tmp / k;
    int half = (int)(k / 2u);
    for (unsigned idx = lane; idx < total; idx += 256u) {
      unsigned item = idx / n;
      unsigned p = idx % n;
      int q = (int)p + (int)kk - half;
      if (q >= 0 && q < (int)n) {
        acc += dz[(item * n + p) * co + o] * x[(item * n + (unsigned)q) * ci + i];
      }
    }
  } else {
    unsigned o = bid - weight_count;
    if (o < co) {
      for (unsigned idx = lane; idx < total; idx += 256u) {
        acc += dz[idx * co + o];
      }
    }
  }
  scratch[lane] = acc;
  __syncthreads();
  for (unsigned step = 128u; step > 0u; step >>= 1) {
    if (lane < step) scratch[lane] += scratch[lane + step];
    __syncthreads();
  }
  if (lane == 0u) {
    if (bid < weight_count) {
      grad[wo + bid] = scratch[0] * inv_batch;
    } else {
      unsigned o = bid - weight_count;
      if (o < co) grad[bo + o] = scratch[0] * inv_batch;
    }
  }
}
