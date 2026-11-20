// Register-blocked 32x32 kernels under test. Host reference lives in test_gemm.cu.
#define TT 32
#define TP 33

__device__ __forceinline__ void feral_split_k_ch(unsigned r, unsigned ch, unsigned ks,
                                                   unsigned *k, unsigned *c) {
  if (ks == 1u) {
    *k = 0u;
    *c = r;
  } else if (ks == 3u) {
    unsigned kk = (unsigned)(r >= ch) + (unsigned)(r >= 2u * ch);
    *k = kk;
    *c = r - kk * ch;
  } else {
    *k = r / ch;
    *c = r % ch;
  }
}

extern "C" __global__ void feral_conv_forward(
    const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ skip,
    float *__restrict__ z, float *__restrict__ y, unsigned frames, unsigned ci, unsigned co,
    unsigned ks, unsigned d, unsigned wo, unsigned bo, unsigned scale, int activation,
    int residual, unsigned batch) {
  const unsigned lc = threadIdx.x, lr = threadIdx.y;
  const unsigned cf = (frames + scale - 1u) / scale;
  const unsigned total = batch * cf;
  const unsigned reduce = ci * ks;
  __shared__ float xs[TT][TP];
  __shared__ float ws[TT][TP];

  const unsigned row0 = lr * 2u, col0 = lc * 2u;          // tile-relative offsets
  const unsigned pg0 = blockIdx.y * TT + row0;
  const unsigned o0 = blockIdx.x * TT + col0;
  const unsigned pg_t[2] = {pg0, pg0 + 1u};
  const unsigned pg_item[2] = {pg_t[0] / cf, pg_t[1] / cf};
  const unsigned pg_q[2] = {pg_t[0] % cf, pg_t[1] % cf};

  float a00 = (o0 < co) ? w[bo + o0] : 0.f;
  float a01 = (o0 + 1u < co) ? w[bo + o0 + 1u] : 0.f;
  float a10 = a00, a11 = a01;

  for (unsigned base = 0; base < reduce; base += TT) {
    for (unsigned t = 0; t < 2u; t++) {
      const unsigned row = row0 + t;
      const unsigned pg = pg_t[t];
      const unsigned item_base = pg_item[t] * cf;
      const unsigned q_base = pg_q[t];
      for (unsigned h = 0; h < 2u; h++) {
        const unsigned col = lc + h * 16u;
        const unsigned r = base + col;
        unsigned k, i;
        feral_split_k_ch(r, ci, ks, &k, &i);
        const int qf = (int)q_base + ((int)k - (int)(ks / 2u)) * (int)d;
        xs[row][col] = (pg < total && r < reduce && qf >= 0 && qf < (int)cf)
                           ? x[(item_base + (unsigned)qf) * ci + i]
                           : 0.f;
        const unsigned o = blockIdx.x * TT + col;
        ws[row][col] = ((base + row) < reduce && o < co)
                           ? w[wo + o * reduce + (base + row)]
                           : 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < TT; j++) {
      const float x0 = xs[row0][j], x1 = xs[row0 + 1u][j];
      const float w0 = ws[j][col0], w1 = ws[j][col0 + 1u];
      a00 += x0 * w0;
      a01 += x0 * w1;
      a10 += x1 * w0;
      a11 += x1 * w1;
    }
    __syncthreads();
  }

#pragma unroll
  for (unsigned ra = 0; ra < 2u; ra++) {
    const unsigned pg = pg_t[ra];
    if (pg >= total) continue;
    const unsigned item = pg_item[ra], q = pg_q[ra];
    const unsigned count = min(scale, frames - q * scale);
    const float v0 = ra ? a10 : a00, v1 = ra ? a11 : a01;
#pragma unroll
    for (unsigned cb = 0; cb < 2u; cb++) {
      const unsigned o = o0 + cb;
      if (o >= co) continue;
      const float act = activation ? tanhf(cb ? v1 : v0) : (cb ? v1 : v0);
      z[pg * co + o] = act;
      for (unsigned r = 0; r < count; r++) {
        const unsigned at = (item * frames + q * scale + r) * co + o;
        y[at] = residual ? skip[at] + 0.5f * act : act;
      }
    }
  }
}

extern "C" __global__ void feral_weight_grad(
    const float *__restrict__ x, const float *__restrict__ dz, float *__restrict__ grad,
    unsigned frames, unsigned ci, unsigned co, unsigned ks, unsigned d, unsigned wo, unsigned bo,
    float inv_batch, unsigned items) {
  const unsigned lc = threadIdx.x, lr = threadIdx.y;
  const unsigned cf = frames, total = items * cf, reduce = ci * ks;
  const unsigned row0 = lr * 2u, col0 = lc * 2u;      // tile-relative: rows = batch pos, cols = r/o
  const unsigned r0 = blockIdx.x * TT + row0;         // output reduction index
  const unsigned o0 = blockIdx.y * TT + col0;         // output channel
  const unsigned r_h[2] = {blockIdx.x * TT + lc, blockIdx.x * TT + lc + 16u};
  unsigned k_h[2], i_h[2];
  feral_split_k_ch(r_h[0], ci, ks, &k_h[0], &i_h[0]);
  feral_split_k_ch(r_h[1], ci, ks, &k_h[1], &i_h[1]);
  __shared__ float zs[TT][TP];  // [batch pos][o tile]
  __shared__ float xs[TT][TP];  // [batch pos][r tile]

  float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f, b0 = 0.f, b1 = 0.f;

  for (unsigned base = 0; base < total; base += TT) {
    for (unsigned t = 0; t < 2u; t++) {
      const unsigned row = row0 + t;
      const unsigned pp = base + row;
      const unsigned q = pp % cf;
      const unsigned item_base = pp - q;
      for (unsigned h = 0; h < 2u; h++) {
        const unsigned col = lc + h * 16u;
        const unsigned o = blockIdx.y * TT + col;
        zs[row][col] = (pp < total && o < co) ? dz[pp * co + o] : 0.f;
        const unsigned r = r_h[h];
        const int qf = (int)q + ((int)k_h[h] - (int)(ks / 2u)) * (int)d;
        xs[row][col] = (pp < total && r < reduce && qf >= 0 && qf < (int)cf)
                           ? x[(item_base + (unsigned)qf) * ci + i_h[h]]
                           : 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < TT; j++) {
      const float x0 = xs[j][row0], x1 = xs[j][row0 + 1u];   // x at reduction r0 / r0+1
      const float z0 = zs[j][col0], z1 = zs[j][col0 + 1u];   // dz at channel o0 / o0+1
      a00 += x0 * z0;
      a01 += x0 * z1;
      a10 += x1 * z0;
      a11 += x1 * z1;
      if (r0 == 0u) {
        b0 += z0;
        b1 += z1;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (unsigned ra = 0; ra < 2u; ra++) {
    const unsigned r = r0 + ra;
    const float v0 = ra ? a10 : a00, v1 = ra ? a11 : a01;
#pragma unroll
    for (unsigned cb = 0; cb < 2u; cb++) {
      const unsigned o = o0 + cb;
      if (r >= reduce || o >= co) continue;
      grad[wo + o * reduce + r] = (cb ? v1 : v0) * inv_batch;
    }
  }
  if (r0 == 0u) {
    if (o0 < co) grad[bo + o0] = b0 * inv_batch;
    if (o0 + 1u < co) grad[bo + o0 + 1u] = b1 * inv_batch;
  }
}

extern "C" __global__ void feral_input_grad(
    const float *__restrict__ dz, const float *__restrict__ w, const float *__restrict__ dh_skip,
    float *__restrict__ dx, unsigned frames, unsigned ci, unsigned co, unsigned ks, unsigned d,
    unsigned wo, unsigned scale, int residual, unsigned items) {
  const unsigned lc = threadIdx.x, lr = threadIdx.y;
  const unsigned cf = (frames + scale - 1u) / scale, total = items * cf, reduce = co * ks;
  const unsigned row0 = lr * 2u, col0 = lc * 2u;   // rows = batch pos (q), cols = input channel
  const unsigned q0 = blockIdx.y * TT + row0;
  const unsigned i0 = blockIdx.x * TT + col0;
  const unsigned qg_t[2] = {q0, q0 + 1u};
  const unsigned qg_item[2] = {qg_t[0] / cf, qg_t[1] / cf};
  const unsigned qg_q[2] = {qg_t[0] % cf, qg_t[1] % cf};
  __shared__ float zs[TT][TP];  // [q][r]
  __shared__ float ws[TT][TP];  // [r][i]

  float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;

  for (unsigned base = 0; base < reduce; base += TT) {
    for (unsigned t = 0; t < 2u; t++) {
      const unsigned row = row0 + t;
      const unsigned qg = qg_t[t];
      const unsigned item_base = qg_item[t] * cf;
      const unsigned q_base = qg_q[t];
      const unsigned r2 = base + row;
      unsigned k2, o2;
      feral_split_k_ch(r2, co, ks, &k2, &o2);
      for (unsigned h = 0; h < 2u; h++) {
        const unsigned col = lc + h * 16u;
        const unsigned rr = base + col;
        unsigned k, o;
        feral_split_k_ch(rr, co, ks, &k, &o);
        const int p = (int)q_base - ((int)k - (int)(ks / 2u)) * (int)d;
        zs[row][col] = (qg < total && rr < reduce && p >= 0 && p < (int)cf)
                           ? dz[(item_base + (unsigned)p) * co + o]
                           : 0.f;
        const unsigned i = blockIdx.x * TT + col;
        ws[row][col] = (r2 < reduce && i < ci) ? w[wo + (o2 * ks + k2) * ci + i] : 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < TT; j++) {
      const float z0 = zs[row0][j], z1 = zs[row0 + 1u][j];
      const float w0 = ws[j][col0], w1 = ws[j][col0 + 1u];
      a00 += z0 * w0;
      a01 += z0 * w1;
      a10 += z1 * w0;
      a11 += z1 * w1;
    }
    __syncthreads();
  }

#pragma unroll
  for (unsigned ra = 0; ra < 2u; ra++) {
    const unsigned qg = qg_t[ra];
    if (qg >= total) continue;
    const float v0 = ra ? a10 : a00, v1 = ra ? a11 : a01;
    const unsigned item = qg_item[ra], q = qg_q[ra];
    const unsigned count = min(scale, frames - q * scale);
#pragma unroll
    for (unsigned cb = 0; cb < 2u; cb++) {
      const unsigned i = i0 + cb;
      if (i >= ci) continue;
      const float v = cb ? v1 : v0;
      for (unsigned r = 0; r < count; r++) {
        const unsigned at = (item * frames + q * scale + r) * ci + i;
        dx[at] = v / (float)count + (residual ? dh_skip[at] : 0.0f);
      }
    }
  }
}
