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
                                           float prior_std, float detail, unsigned items,
                                           unsigned audio_ch) {
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
    // Detail term: mean squared first difference of the residual per audio channel over pairs
    // where both samples are unknown, so known regions never leak in.
    if (detail > 0.f) {
      unsigned base = item * n, end = base + n;
      if (m == 0.f && id + audio_ch < end && data[2u * total + id + audio_ch] == 0.f) {
        float delta =
            feral_residual(prediction, data, total, id + audio_ch, pre, t, cskip, cout) - r;
        value += detail * delta * delta / u;
        dyv -= 2.f * detail * delta / u;
      }
      if (m == 0.f && id >= base + audio_ch && data[2u * total + id - audio_ch] == 0.f) {
        float delta =
            r - feral_residual(prediction, data, total, id - audio_ch, pre, t, cskip, cout);
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
                                      float *__restrict__ stats, const float *__restrict__ decay_mask,
                                      unsigned n, float lr, float c1, float c2, float wd) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !isfinite(stats[0]) || !isfinite(stats[1])) return;
  float g = grad[i] * stats[2], mm = 0.9f * m[i] + 0.1f * g,
        vv = 0.999f * v[i] + 0.001f * g * g;
  float w = weights[i];
  if (wd > 0.0f) w *= (1.0f - lr * wd * decay_mask[i]);
  float p = w - lr * (mm / c1) / (sqrtf(vv / c2) + 1e-8f);
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
                                                unsigned block, unsigned audio_ch,
                                                unsigned items) {
  unsigned total = items * frames * block;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  unsigned p = idx / block;
  unsigned c = idx % block;
  unsigned frame_elems = block * audio_ch;
  unsigned channels = 3u * frame_elems + 4u;
  unsigned base = p * channels;
  unsigned dst = idx * (3u * audio_ch);
  for (unsigned ch = 0u; ch < audio_ch; ch++) {
    stem_in[dst + ch] = x_features[base + c * audio_ch + ch];
    stem_in[dst + audio_ch + ch] = x_features[base + frame_elems + c * audio_ch + ch];
    stem_in[dst + 2u * audio_ch + ch] = x_features[base + 2u * frame_elems + c * audio_ch + ch];
  }
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

// ============================================================================
// Gated-FiLM residual block kernels (pre-RMSNorm + per-block FiLM + tanh*sigmoid)
// ============================================================================

extern "C" __global__ void feral_time_embed_proj_fwd(
    const float *__restrict__ data, const float *__restrict__ weights, float *__restrict__ film,
    unsigned total_samples, unsigned width, unsigned wo, unsigned bo, unsigned items) {
  unsigned two_w = 2u * width;
  unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned item = blockIdx.y;
  if (o >= two_w || item >= items) return;
  float t = data[3u * total_samples + 4u * item];
  float sum = weights[bo + o];
  for (unsigned k = 0u; k < 16u; k++) {
    float freq = 6.2831853071795864769f * powf(2.0f, (float)k * 0.5f);
    float ang = t * freq;
    sum += weights[wo + o * 32u + 2u * k] * sinf(ang) +
           weights[wo + o * 32u + 2u * k + 1u] * cosf(ang);
  }
  film[item * two_w + o] = sum;
}

extern "C" __global__ void feral_rmsnorm_film_fwd(
    const float *__restrict__ h, const float *__restrict__ film, float *__restrict__ normed,
    float *__restrict__ inv_rms_out, unsigned frames, unsigned width, unsigned items) {
  unsigned pf = blockIdx.x;
  if (pf >= items * frames) return;
  unsigned item = pf / frames;
  unsigned lane = threadIdx.x;
  __shared__ float scratch[256];
  const float *row = h + pf * width;
  float sq = 0.0f;
  for (unsigned c = lane; c < width; c += 256u) {
    float v = row[c];
    sq += v * v;
  }
  scratch[lane] = sq;
  __syncthreads();
  for (unsigned step = 128u; step > 0u; step >>= 1) {
    if (lane < step) scratch[lane] += scratch[lane + step];
    __syncthreads();
  }
  float inv_rms = 1.0f / sqrtf(scratch[0] / (float)width + 1e-5f);
  if (lane == 0u) inv_rms_out[pf] = inv_rms;
  const float *f = film + item * 2u * width;
  float *dst = normed + pf * width;
  for (unsigned c = lane; c < width; c += 256u) {
    dst[c] = (1.0f + f[c]) * (row[c] * inv_rms) + f[width + c];
  }
}

extern "C" __global__ void feral_gated_residual_fwd(
    float *__restrict__ z, const float *__restrict__ skip, float *__restrict__ y, unsigned frames,
    unsigned width, unsigned scale, int is_glu, float res_scale, unsigned items) {
  unsigned cf = (frames + scale - 1u) / scale;
  unsigned total = items * cf * width;
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= total) return;
  unsigned c = id % width;
  unsigned pg = id / width;
  unsigned item = pg / cf;
  unsigned q = pg % cf;
  unsigned base = pg * 2u * width;
  float raw_a = z[base + c];
  float ta = is_glu ? raw_a : tanhf(raw_a);
  float sb = 1.0f / (1.0f + expf(-z[base + width + c]));
  z[base + c] = ta;
  z[base + width + c] = sb;
  float act = ta * sb;
  unsigned count = min(scale, frames - q * scale);
  for (unsigned r = 0u; r < count; r++) {
    unsigned at = (item * frames + q * scale + r) * width + c;
    y[at] = skip[at] + res_scale * act;
  }
}

extern "C" __global__ void feral_gated_activation_grad(
    const float *__restrict__ dh, const float *__restrict__ z, float *__restrict__ dz,
    unsigned frames, unsigned width, unsigned scale, int is_glu, float res_scale,
    unsigned items) {
  unsigned cf = (frames + scale - 1u) / scale;
  unsigned total = items * cf * width;
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= total) return;
  unsigned c = id % width;
  unsigned pg = id / width;
  unsigned item = pg / cf;
  unsigned q = pg % cf;
  unsigned count = min(scale, frames - q * scale);
  float g = 0.0f;
  for (unsigned r = 0u; r < count; r++) {
    g += dh[(item * frames + q * scale + r) * width + c];
  }
  g *= res_scale;
  unsigned base = pg * 2u * width;
  float ta = z[base + c];
  float sb = z[base + width + c];
  dz[base + c] = is_glu ? (g * sb) : (g * sb * (1.0f - ta * ta));
  dz[base + width + c] = g * ta * sb * (1.0f - sb);
}

extern "C" __global__ void feral_rmsnorm_film_dx(
    const float *__restrict__ du, const float *__restrict__ h, const float *__restrict__ inv_rms,
    const float *__restrict__ film, const float *__restrict__ dh_skip, float *__restrict__ dh_out,
    unsigned frames, unsigned width, unsigned items) {
  unsigned pf = blockIdx.x;
  if (pf >= items * frames) return;
  unsigned item = pf / frames;
  unsigned lane = threadIdx.x;
  __shared__ float scratch[256];
  float r = inv_rms[pf];
  const float *h_row = h + pf * width;
  const float *du_row = du + pf * width;
  const float *f = film + item * 2u * width;
  float dot_acc = 0.0f;
  for (unsigned c = lane; c < width; c += 256u) {
    float xh = h_row[c] * r;
    float gt = (1.0f + f[c]) * du_row[c];
    dot_acc += gt * xh;
  }
  scratch[lane] = dot_acc;
  __syncthreads();
  for (unsigned step = 128u; step > 0u; step >>= 1) {
    if (lane < step) scratch[lane] += scratch[lane + step];
    __syncthreads();
  }
  float dot = scratch[0] / (float)width;
  const float *skip_row = dh_skip + pf * width;
  float *out_row = dh_out + pf * width;
  for (unsigned c = lane; c < width; c += 256u) {
    float xh = h_row[c] * r;
    float gt = (1.0f + f[c]) * du_row[c];
    out_row[c] = r * (gt - xh * dot) + skip_row[c];
  }
}

extern "C" __global__ void feral_rmsnorm_film_dfilm(
    const float *__restrict__ du, const float *__restrict__ h, const float *__restrict__ inv_rms,
    float *__restrict__ d_film, unsigned frames, unsigned width, unsigned items) {
  unsigned o = blockIdx.x;
  unsigned item = blockIdx.y;
  if (o >= 2u * width || item >= items) return;
  unsigned lane = threadIdx.x;
  __shared__ float scratch[256];
  float acc = 0.0f;
  unsigned base_f = item * frames;
  if (o < width) {
    for (unsigned p = lane; p < frames; p += 256u) {
      unsigned pf = base_f + p;
      acc += du[pf * width + o] * h[pf * width + o] * inv_rms[pf];
    }
  } else {
    unsigned c = o - width;
    for (unsigned p = lane; p < frames; p += 256u) {
      unsigned pf = base_f + p;
      acc += du[pf * width + c];
    }
  }
  scratch[lane] = acc;
  __syncthreads();
  for (unsigned step = 128u; step > 0u; step >>= 1) {
    if (lane < step) scratch[lane] += scratch[lane + step];
    __syncthreads();
  }
  if (lane == 0u) d_film[item * 2u * width + o] = scratch[0];
}

extern "C" __global__ void feral_time_proj_dw(
    const float *__restrict__ data, const float *__restrict__ d_film, float *__restrict__ grad,
    unsigned total_samples, unsigned width, unsigned wo, unsigned bo, float inv_batch,
    unsigned items) {
  unsigned o = blockIdx.x;
  unsigned i = threadIdx.x;
  if (o >= 2u * width || i >= 32u) return;
  unsigned k = i >> 1;
  float freq = 6.2831853071795864769f * powf(2.0f, (float)k * 0.5f);
  float w_acc = 0.0f;
  float b_acc = 0.0f;
  for (unsigned item = 0u; item < items; item++) {
    float t = data[3u * total_samples + 4u * item];
    float te_i = (i & 1u) ? cosf(t * freq) : sinf(t * freq);
    float df = d_film[item * 2u * width + o];
    w_acc += df * te_i;
    if (i == 0u) b_acc += df;
  }
  grad[wo + o * 32u + i] = w_acc * inv_batch;
  if (i == 0u) grad[bo + o] = b_acc * inv_batch;
}

// ============================================================================
// GPU-resident multiresolution STFT log-magnitude Charbonnier L1 endpoint loss
// Zero PCIe sync, zero atomics (4 non-overlapping hop=N/4 slots -> bit-exact).
// ============================================================================

extern "C" __global__ void feral_zero_buf(float *__restrict__ dst, unsigned total) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total) dst[id] = 0.0f;
}

__device__ __forceinline__ void feral_radix2_fft(float *s_re, float *s_im, unsigned size,
                                                 float sign) {
  unsigned tid = threadIdx.x;
  unsigned nthreads = blockDim.x;
  unsigned half_total = size >> 1;
  for (unsigned len = 2u; len <= size; len <<= 1) {
    unsigned half = len >> 1;
    float angle_step = sign * 6.2831853071795864769f / (float)len;
    for (unsigned b = tid; b < half_total; b += nthreads) {
      unsigned group = b / half;
      unsigned k = b - group * half;
      unsigned i0 = group * len + k;
      unsigned i1 = i0 + half;
      float ang = angle_step * (float)k;
      float wr = cosf(ang);
      float wi = sinf(ang);
      float u_re = s_re[i0], u_im = s_im[i0];
      float v_re = s_re[i1] * wr - s_im[i1] * wi;
      float v_im = s_re[i1] * wi + s_im[i1] * wr;
      s_re[i0] = u_re + v_re;
      s_im[i0] = u_im + v_im;
      s_re[i1] = u_re - v_re;
      s_im[i1] = u_im - v_im;
    }
    __syncthreads();
  }
}

extern "C" __global__ void feral_stft_frame(
    const float *__restrict__ pred, const float *__restrict__ data, float *__restrict__ overlap,
    float *__restrict__ frame_losses, unsigned n, unsigned items, unsigned size, unsigned log2_n,
    unsigned hop, unsigned frame_count, int pre, float data_scale, float prior_std,
    float spectral_weight, float num_resolutions, unsigned audio_ch, int mode, float num_streams) {
  unsigned frame_idx = blockIdx.x;
  unsigned by = blockIdx.y;
  unsigned item = by / audio_ch;
  unsigned ch = by % audio_ch;
  if (frame_idx >= frame_count || item >= items) return;
  unsigned tid = threadIdx.x;
  unsigned nthreads = blockDim.x;
  unsigned total = items * n * audio_ch;
  float t = data[3u * total + 4u * item];
  if (spectral_weight <= 0.0f || t <= 0.0f || t >= 1.0f) {
    if (tid == 0u) frame_losses[by * frame_count + frame_idx] = 0.0f;
    return;
  }

  __shared__ float s_re[2048];
  __shared__ float s_im[2048];
  __shared__ float s_loss[256];

  float cskip = 0.0f, cout = 1.0f;
  if (pre) feral_precond_coeffs(t, data_scale, prior_std, &cskip, &cout);
  float omt = 1.0f - t;
  unsigned start = frame_idx * hop;
  unsigned len = (start < n) ? min(size, n - start) : 0u;
  float inv_size_tau = 6.2831853071795864769f / (float)size;
  const float inv_sqrt2 = 0.7071067811865475f;

  // Pack real yhat_w into Re and real clean_w into Im (exact 1-FFT real-pair transform).
  for (unsigned j = tid; j < size; j += nthreads) {
    float yhw = 0.0f, yw = 0.0f;
    if (j < len) {
      float y, yh;
      if (mode == 0) {
        unsigned idx = (item * n + start + j) * audio_ch + ch;
        y = data[idx];
        float z = data[total + idx], m = data[2u * total + idx];
        if (m == 1.0f) {
          yh = y;
        } else {
          float xt = omt * z + t * y;
          float v = cskip * xt + cout * pred[idx];
          yh = xt + omt * v;
        }
      } else {
        unsigned idxL = (item * n + start + j) * 2u;
        unsigned idxR = idxL + 1u;
        float yL = data[idxL], zL = data[total + idxL], mL = data[2u * total + idxL];
        float yR = data[idxR], zR = data[total + idxR], mR = data[2u * total + idxR];
        float yhL, yhR;
        if (mL == 1.0f) {
          yhL = yL;
        } else {
          float xtL = omt * zL + t * yL;
          float vL = cskip * xtL + cout * pred[idxL];
          yhL = xtL + omt * vL;
        }
        if (mR == 1.0f) {
          yhR = yR;
        } else {
          float xtR = omt * zR + t * yR;
          float vR = cskip * xtR + cout * pred[idxR];
          yhR = xtR + omt * vR;
        }
        y = (ch == 0u) ? inv_sqrt2 * (yL + yR) : inv_sqrt2 * (yL - yR);
        yh = (ch == 0u) ? inv_sqrt2 * (yhL + yhR) : inv_sqrt2 * (yhL - yhR);
      }
      float w = 0.5f - 0.5f * cosf(inv_size_tau * (float)j);
      yhw = yh * w;
      yw = y * w;
    }
    unsigned br = __brev(j) >> (32u - log2_n);
    s_re[br] = yhw;
    s_im[br] = yw;
  }
  __syncthreads();

  feral_radix2_fft(s_re, s_im, size, -1.0f);

  unsigned bins = (size >> 1) + 1u;
  float scale = 1.0f / (num_streams * num_resolutions * (float)(frame_count * bins));
  const float eps_sq = 1e-10f;
  const float delta = 1e-5f;
  const float delta_sq = 1e-10f;
  float g_re[5], g_im[5];
  float local_loss = 0.0f;
  unsigned idx_local = 0u;
  for (unsigned k = tid; k < bins; k += nthreads, idx_local++) {
    unsigned km = (k == 0u) ? 0u : (size - k);
    float zk_re = s_re[k], zk_im = s_im[k];
    float zm_re = s_re[km], zm_im = s_im[km];
    float a_re = 0.5f * (zk_re + zm_re);
    float a_im = 0.5f * (zk_im - zm_im);
    float b_re = 0.5f * (zk_im + zm_im);
    float b_im = 0.5f * (zm_re - zk_re);
    float mag_a = sqrtf(a_re * a_re + a_im * a_im + eps_sq);
    float mag_b = sqrtf(b_re * b_re + b_im * b_im + eps_sq);
    float diff = log1pf(mag_a) - log1pf(mag_b);
    float rho = sqrtf(diff * diff + delta_sq);
    local_loss += (rho - delta);
    float coeff = scale * (diff / rho) / ((1.0f + mag_a) * mag_a);
    g_re[idx_local] = a_re * coeff;
    g_im[idx_local] = a_im * coeff;
  }
  __syncthreads();

  idx_local = 0u;
  for (unsigned k = tid; k < size; k += nthreads, idx_local++) {
    unsigned br = __brev(k) >> (32u - log2_n);
    if (k < bins) {
      s_re[br] = g_re[idx_local];
      s_im[br] = g_im[idx_local];
    } else {
      s_re[br] = 0.0f;
      s_im[br] = 0.0f;
    }
  }
  __syncthreads();

  feral_radix2_fft(s_re, s_im, size, 1.0f);

  float weight = spectral_weight * t;
  float grad_scale = weight * omt * cout;
  unsigned slot_base = (frame_idx & 3u) * total + (item * n + start) * audio_ch;
  for (unsigned j = tid; j < len; j += nthreads) {
    float w = 0.5f - 0.5f * cosf(inv_size_tau * (float)j);
    if (mode == 0) {
      unsigned idx = (item * n + start + j) * audio_ch + ch;
      if (data[2u * total + idx] == 0.0f) {
        overlap[slot_base + j * audio_ch + ch] = w * s_re[j] * grad_scale;
      }
    } else {
      overlap[slot_base + j * 2u + ch] = w * s_re[j] * grad_scale;
    }
  }

  s_loss[tid] = local_loss;
  __syncthreads();
  for (unsigned step = nthreads >> 1; step > 0u; step >>= 1) {
    if (tid < step) s_loss[tid] += s_loss[tid + step];
    __syncthreads();
  }
  if (tid == 0u) frame_losses[by * frame_count + frame_idx] = s_loss[0] * scale * weight;
}

extern "C" __global__ void feral_stft_accum(float *__restrict__ dy,
                                            const float *__restrict__ overlap, unsigned total) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total) {
    dy[id] += overlap[id] + overlap[total + id] + overlap[2u * total + id] +
              overlap[3u * total + id];
  }
}

extern "C" __global__ void feral_stft_accum_ms(float *__restrict__ dy,
                                               const float *__restrict__ overlap,
                                               const float *__restrict__ data, unsigned total_time,
                                               unsigned total_scalars) {
  unsigned ti = blockIdx.x * blockDim.x + threadIdx.x;
  if (ti >= total_time) return;
  unsigned iL = ti * 2u;
  unsigned iR = iL + 1u;
  float gM = overlap[iL] + overlap[total_scalars + iL] + overlap[2u * total_scalars + iL] +
             overlap[3u * total_scalars + iL];
  float gS = overlap[iR] + overlap[total_scalars + iR] + overlap[2u * total_scalars + iR] +
             overlap[3u * total_scalars + iR];
  const float inv_s2 = 0.7071067811865475f;
  if (data[2u * total_scalars + iL] == 0.0f) dy[iL] += inv_s2 * (gM + gS);
  if (data[2u * total_scalars + iR] == 0.0f) dy[iR] += inv_s2 * (gM - gS);
}

extern "C" __global__ void feral_add_loss_sum(float *__restrict__ loss_parts,
                                              const float *__restrict__ frame_losses,
                                              unsigned count) {
  unsigned lane = threadIdx.x;
  __shared__ float scratch[256];
  float acc = 0.0f;
  for (unsigned i = lane; i < count; i += 256u) acc += frame_losses[i];
  scratch[lane] = acc;
  __syncthreads();
  for (unsigned step = 128u; step > 0u; step >>= 1) {
    if (lane < step) scratch[lane] += scratch[lane + step];
    __syncthreads();
  }
  if (lane == 0u) loss_parts[0] += scratch[0];
}
