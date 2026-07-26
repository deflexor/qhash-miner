/**
 * qhash CUDA kernels.
 *
 * qhash_cf_kernel (Phase 6) is the mining path: the closed-form ⟨Z⟩ sweep, one
 * nonce per thread, no state buffer and no per-nonce barrier. Everything a nonce
 * needs lives in registers plus a small angle table in shared memory.
 *
 * qhash_mine_kernel is the original 65536-amplitude statevector, one block per
 * nonce, kept as the consensus oracle: the self-test, the cuStateVec golden
 * harness and candidate re-verification all run through it. It is no longer
 * performance-relevant, so Phase 6.13 stripped the tiling, fiber-blocking,
 * double-buffering and boundary-pairing variants that used to surround it.
 *
 * When compiled without CUDA (QHASH_CPU_ONLY), falls back to the CPU simulator.
 */
#include "qhash_kernel.cuh"
#include "circuit.cuh"
#include "closed_form.cuh"
#include "sha256.cuh"
#include "sha256_mine.cuh"
#include "target.cuh"
#include "qhash_cpu.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

using namespace qhash;

#ifndef QHASH_CPU_ONLY
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err_ = (x);                                                \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(err_), __FILE__, __LINE__);        \
            return -1;                                                         \
        }                                                                      \
    } while (0)

/* ---------- device helpers ---------- */

template <typename Real>
__device__ void d_init_zero(Complex<Real>* psi, int tid, int nthreads)
{
    for (int i = tid; i < QHASH_STATE_SIZE; i += nthreads)
        psi[i] = Complex<Real>(0, 0);
    __syncthreads();
    if (tid == 0)
        psi[0] = Complex<Real>(1, 0);
    __syncthreads();
}

/**
 * Apply 2×2 on qubit q over an n-amplitude buffer (n power-of-two, q < log2(n)).
 * Used for both full state (global) and shared-memory tiles.
 * Two-pair ILP step when the grid covers ≥2×nthreads pairs.
 */
template <typename Real>
__device__ void d_apply_u2_n(Complex<Real>* psi, int q, const U2<Real>& u, int tid, int nthreads,
                             int n)
{
    const int mask = 1 << q;
    const int npairs = n >> 1;
    const int step = nthreads * 2;
    int p = tid;
    for (; p + nthreads < npairs; p += step) {
        const int p1 = p;
        const int p2 = p + nthreads;
        const int lo1 = p1 & ((1 << q) - 1);
        const int hi1 = p1 >> q;
        const int i1 = lo1 | (hi1 << (q + 1));
        const int j1 = i1 | mask;
        const int lo2 = p2 & ((1 << q) - 1);
        const int hi2 = p2 >> q;
        const int i2 = lo2 | (hi2 << (q + 1));
        const int j2 = i2 | mask;
        Complex<Real> a1 = psi[i1];
        Complex<Real> b1 = psi[j1];
        Complex<Real> a2 = psi[i2];
        Complex<Real> b2 = psi[j2];
        apply_u2_pair(a1, b1, u);
        apply_u2_pair(a2, b2, u);
        psi[i1] = a1;
        psi[j1] = b1;
        psi[i2] = a2;
        psi[j2] = b2;
    }
    for (; p < npairs; p += nthreads) {
        const int lo = p & ((1 << q) - 1);
        const int hi = p >> q;
        const int i = lo | (hi << (q + 1));
        const int j = i | mask;
        Complex<Real> a = psi[i];
        Complex<Real> b = psi[j];
        apply_u2_pair(a, b, u);
        psi[i] = a;
        psi[j] = b;
    }
    __syncthreads();
}

template <typename Real>
__device__ void d_apply_u2(Complex<Real>* psi, int q, const U2<Real>& u, int tid, int nthreads)
{
    d_apply_u2_n(psi, q, u, tid, nthreads, QHASH_STATE_SIZE);
}

/**
 * CNOT via pair-index tiling (Phase 2.2).
 *
 * Only n/4 swaps are live (control=1, target bit flips). Map pair index p the
 * same way as d_apply_u2, inserting gaps at both qubit positions, then set
 * control=1. Avoids full-state scan + divergent branch of the Phase-1 loop.
 */
template <typename Real>
__device__ void d_apply_cnot_n(Complex<Real>* psi, int control, int target, int tid, int nthreads,
                               int n)
{
    const int npairs = n >> 2;
    const int cmask = 1 << control;
    const int tmask = 1 << target;
    const int q0 = control < target ? control : target;
    const int q1 = control < target ? target : control;
    const int lo_bits = q0;
    const int mid_bits = q1 - q0 - 1;
    const int lo_mask = (1 << lo_bits) - 1;
    const int mid_mask = (1 << mid_bits) - 1;

    const int step = nthreads * 2;
    int p = tid;
    for (; p + nthreads < npairs; p += step) {
        const int p1 = p;
        const int p2 = p + nthreads;
        const int lo1 = p1 & lo_mask;
        const int mid1 = (p1 >> lo_bits) & mid_mask;
        const int hi1 = p1 >> (lo_bits + mid_bits);
        const int base1 = lo1 | (mid1 << (q0 + 1)) | (hi1 << (q1 + 1));
        const int i1 = base1 | cmask;
        const int j1 = i1 | tmask;
        const int lo2 = p2 & lo_mask;
        const int mid2 = (p2 >> lo_bits) & mid_mask;
        const int hi2 = p2 >> (lo_bits + mid_bits);
        const int base2 = lo2 | (mid2 << (q0 + 1)) | (hi2 << (q1 + 1));
        const int i2 = base2 | cmask;
        const int j2 = i2 | tmask;
        const Complex<Real> t1 = psi[i1];
        const Complex<Real> t2 = psi[i2];
        psi[i1] = psi[j1];
        psi[j1] = t1;
        psi[i2] = psi[j2];
        psi[j2] = t2;
    }
    for (; p < npairs; p += nthreads) {
        const int lo = p & lo_mask;
        const int mid = (p >> lo_bits) & mid_mask;
        const int hi = p >> (lo_bits + mid_bits);
        const int base = lo | (mid << (q0 + 1)) | (hi << (q1 + 1));
        const int i = base | cmask;
        const int j = i | tmask;
        const Complex<Real> tmp = psi[i];
        psi[i] = psi[j];
        psi[j] = tmp;
    }
    __syncthreads();
}

template <typename Real>
__device__ void d_apply_cnot(Complex<Real>* psi, int control, int target, int tid, int nthreads)
{
    d_apply_cnot_n(psi, control, target, tid, nthreads, QHASH_STATE_SIZE);
}

/**
 * The circuit, one gate at a time, straight over the state in global memory.
 *
 * This kernel is now only the consensus oracle: the Phase-6 closed form took over
 * mining, so the shared-memory tiling, high-qubit fiber blocking, tile
 * double-buffering and boundary pairing that Phases 2-3b grew around it were
 * retired in 6.13 along with their build options. What is left applies the gates
 * in exactly the order circuit.cuh documents, which is the version worth trusting
 * and the one the CPU reference matches gate for gate.
 */
template <typename Real>
__device__ void d_run_circuit(Complex<Real>* psi, const uint8_t* nibbles, int offset, int tid,
                              int nthreads)
{
    for (int l = 0; l < QHASH_NUM_LAYERS; ++l) {
        for (int i = 0; i < QHASH_NUM_QUBITS; ++i) {
            const uint8_t ny = nibbles[(2 * l * QHASH_NUM_QUBITS + i) % QHASH_NIBBLE_COUNT];
            const uint8_t nz = nibbles[((2 * l + 1) * QHASH_NUM_QUBITS + i) % QHASH_NIBBLE_COUNT];
            const Real ty = nibble_angle<Real>(ny, offset);
            const Real tz = nibble_angle<Real>(nz, offset);
#ifdef QHASH_FUSE_RZ_RY
            d_apply_u2(psi, i, make_rz_ry<Real>(ty, tz), tid, nthreads);
#else
            d_apply_u2(psi, i, make_ry<Real>(ty), tid, nthreads);
            d_apply_u2(psi, i, make_rz<Real>(tz), tid, nthreads);
#endif
        }
        for (int i = 0; i < QHASH_NUM_QUBITS - 1; ++i)
            d_apply_cnot(psi, i, i + 1, tid, nthreads);
    }
}

template <typename Real>
__device__ void d_expectations(const Complex<Real>* psi, Real* exp_out, int tid, int nthreads)
{
    __shared__ Real buf[QHASH_MAX_THREADS];
    Real local[QHASH_NUM_QUBITS];
#pragma unroll
    for (int q = 0; q < QHASH_NUM_QUBITS; ++q)
        local[q] = Real(0);

    for (int i = tid; i < QHASH_STATE_SIZE; i += nthreads) {
        const Real p = psi[i].norm2();
#pragma unroll
        for (int q = 0; q < QHASH_NUM_QUBITS; ++q)
            local[q] += (i & (1 << q)) ? -p : p;
    }

    for (int q = 0; q < QHASH_NUM_QUBITS; ++q) {
        buf[tid] = local[q];
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s)
                buf[tid] += buf[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            exp_out[q] = buf[0];
        __syncthreads();
    }
}

__device__ int d_hash_le_target(const uint8_t hash[32], const uint8_t target[32])
{
    return qhash_hash_le_target(hash, target);
}

template <typename Real>
__global__ void qhash_mine_kernel(const uint8_t* header_template,
                                  uint32_t nonce_start,
                                  uint32_t nTime,
                                  int offset,
                                  const uint8_t* target,
                                  int check_target,
                                  Complex<Real>* states, /* [grid][STATE] */
                                  qhash_share_t* shares,
                                  uint32_t* share_count,
                                  uint32_t max_shares,
                                  uint8_t* hash_scratch) /* optional debug: [grid][32] */
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const uint32_t nonce = nonce_start + uint32_t(bid);

    Complex<Real>* psi = states + size_t(bid) * QHASH_STATE_SIZE;

    __shared__ uint8_t s_header[QHASH_INPUT_SIZE];
    __shared__ uint8_t s_hash[QHASH_SHA256_SIZE];
    __shared__ uint8_t s_nibbles[QHASH_NIBBLE_COUNT];
    __shared__ Real s_exp[QHASH_NUM_QUBITS];
    __shared__ uint8_t s_buf[QHASH_SHA256_SIZE + QHASH_NUM_QUBITS * 2];
    __shared__ uint8_t s_out[QHASH_SHA256_SIZE];

    if (tid < QHASH_INPUT_SIZE)
        s_header[tid] = header_template[tid];
    __syncthreads();
    if (tid == 0) {
        s_header[76] = uint8_t(nonce);
        s_header[77] = uint8_t(nonce >> 8);
        s_header[78] = uint8_t(nonce >> 16);
        s_header[79] = uint8_t(nonce >> 24);
    }
    __syncthreads();

    if (tid == 0)
        sha256(s_header, QHASH_INPUT_SIZE, s_hash);
    __syncthreads();

    if (tid == 0)
        split_nibbles(s_hash, s_nibbles);
    __syncthreads();

    d_init_zero(psi, tid, nthreads);
    d_run_circuit(psi, s_nibbles, offset, tid, nthreads);
    d_expectations(psi, s_exp, tid, nthreads);

    if (tid == 0) {
        for (int i = 0; i < QHASH_SHA256_SIZE; ++i)
            s_buf[i] = s_hash[i];
        int zeroes = 0;
        for (int i = 0; i < QHASH_NUM_QUBITS; ++i) {
            const int16_t fixed = to_fixed_q15(double(s_exp[i]));
            const int j = QHASH_SHA256_SIZE + i * 2;
            s_buf[j] = uint8_t(fixed & 0xFF);
            s_buf[j + 1] = uint8_t((fixed >> 8) & 0xFF);
            if (s_buf[j] == 0)
                ++zeroes;
            if (s_buf[j + 1] == 0)
                ++zeroes;
        }
        if (softfork_reject_zeroes(nTime, zeroes)) {
            for (int i = 0; i < QHASH_SHA256_SIZE; ++i)
                s_out[i] = 0xFF;
        } else {
            sha256(s_buf, sizeof(s_buf), s_out);
        }

        if (hash_scratch) {
            for (int i = 0; i < 32; ++i)
                hash_scratch[bid * 32 + i] = s_out[i];
        }

        int ok = 1;
        if (check_target)
            ok = d_hash_le_target(s_out, target);
        if (ok) {
            const uint32_t slot = atomicAdd(share_count, 1u);
            if (slot < max_shares) {
                shares[slot].nonce = nonce;
                for (int i = 0; i < 32; ++i)
                    shares[slot].hash[i] = s_out[i];
            }
        }
    }
}

/* ======================================================================== *
 * Phase 6: closed-form kernel — one nonce per thread                       *
 *                                                                          *
 * No state buffer, no per-nonce barrier, no global traffic beyond the share *
 * queue. Everything a nonce needs is in registers plus a 512-byte angle     *
 * table staged into shared memory once per block.                          *
 * ======================================================================== */

/**
 * Per-job constants. All of it is read with a warp-uniform index, so constant
 * memory broadcasts and costs nothing — except the angle table, whose index is a
 * data-dependent nibble, so that one is staged into shared memory instead.
 */
struct CfParams {
    uint32_t midstate[8]; /* SHA-256 state after header bytes 0..63 */
    uint32_t hdr_w[3];    /* header bytes 64..75 as big-endian schedule words */
    uint32_t pad_w[64];   /* constant schedule of the final hash's padding block */
    uint32_t target[8];   /* target as big-endian words */
    AngleLut lut;         /* built with host libm, so host and device agree exactly */
    uint32_t nTime;
    int check_target;
};

__constant__ CfParams c_cf;

/** Byte-reverse, turning the little-endian header nonce into its schedule word. */
__device__ __forceinline__ uint32_t cf_bswap32(uint32_t x)
{
    return ((x & 0x000000FFu) << 24) | ((x & 0x0000FF00u) << 8) | ((x & 0x00FF0000u) >> 8) |
           ((x & 0xFF000000u) >> 24);
}

__device__ __forceinline__ bool cf_le_target(const uint32_t h[8], const uint32_t t[8])
{
    return qhash_digest_le_target(h, t);
}

__global__ void qhash_cf_kernel(uint32_t nonce_start, uint32_t nonce_count, qhash_share_t* shares,
                                uint32_t* share_count, uint32_t max_shares)
{
    /* One barrier for the entire kernel: the grid-stride loop below runs many
       nonces per thread, so this is amortised away rather than paid per nonce. */
    __shared__ AngleLut s_lut;
    {
        double* dst = reinterpret_cast<double*>(&s_lut);
        const double* src = reinterpret_cast<const double*>(&c_cf.lut);
        for (int i = int(threadIdx.x); i < QHASH_ANGLE_LUT_DOUBLES; i += int(blockDim.x))
            dst[i] = src[i];
    }
    __syncthreads();

    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nonce_count; idx += stride) {
        const uint32_t nonce = nonce_start + idx;

        /* --- SHA256(header): job midstate + the one nonce-dependent block --- */
        uint32_t hs[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            hs[i] = c_cf.midstate[i];

        uint32_t w[16];
        w[0] = c_cf.hdr_w[0];
        w[1] = c_cf.hdr_w[1];
        w[2] = c_cf.hdr_w[2];
        w[3] = cf_bswap32(nonce); /* header bytes 76..79 */
        w[4] = 0x80000000u;       /* padding: W[4..15] are constants */
#pragma unroll
        for (int i = 5; i < 15; ++i)
            w[i] = 0;
        w[15] = QHASH_INPUT_SIZE * 8;
        sha256_compress(hs, w);

        /* --- closed-form ⟨Z⟩ straight out of the header hash words --- */
        double e[QHASH_NUM_QUBITS];
        closed_form_sweep(s_lut, NibbleWords{hs}, e);

        /* --- Q1.15 little-endian pairs → schedule words 8..15, counting zeros --- */
        uint32_t fw[16];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            fw[i] = hs[i];

        int zeroes = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int16_t f0 = to_fixed_q15(e[2 * i]);
            const int16_t f1 = to_fixed_q15(e[2 * i + 1]);
            const uint32_t b0 = uint32_t(f0) & 0xFFu;
            const uint32_t b1 = (uint32_t(f0) >> 8) & 0xFFu;
            const uint32_t b2 = uint32_t(f1) & 0xFFu;
            const uint32_t b3 = (uint32_t(f1) >> 8) & 0xFFu;
            fw[8 + i] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
            zeroes += int(b0 == 0) + int(b1 == 0) + int(b2 == 0) + int(b3 == 0);
        }

        uint32_t digest[8];
        if (softfork_reject_zeroes(c_cf.nTime, zeroes)) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                digest[i] = 0xFFFFFFFFu;
        } else {
            sha256_init(digest);
            sha256_compress(digest, fw);
            sha256_compress_const(digest, c_cf.pad_w);
        }

        if (c_cf.check_target && !cf_le_target(digest, c_cf.target))
            continue;

        const uint32_t slot = atomicAdd(share_count, 1u);
        if (slot < max_shares) {
            shares[slot].nonce = nonce;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                shares[slot].hash[4 * i + 0] = uint8_t(digest[i] >> 24);
                shares[slot].hash[4 * i + 1] = uint8_t(digest[i] >> 16);
                shares[slot].hash[4 * i + 2] = uint8_t(digest[i] >> 8);
                shares[slot].hash[4 * i + 3] = uint8_t(digest[i]);
            }
        }
    }
}

extern "C" int qhash_cuda_available(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
        return 0;
    return n > 0;
}

extern "C" int qhash_cuda_sm_count(void)
{
    /* cudaGetDeviceProperties returns a struct whose layout moves between
       toolkit versions; the attribute query is ABI-stable and is the only one
       that reports the real count when the build and runtime headers differ. */
    static int cached = -1;
    if (cached < 0) {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            cached = 0;
        else
            cached = n;
    }
    return cached;
}

namespace {

/**
 * Device buffers for the share queue, kept alive across calls. The closed-form
 * kernel finishes a batch in a single launch, so per-call cudaMalloc/cudaFree
 * would be a visible fraction of the runtime.
 */
struct CfBuffers {
    std::mutex mu;
    qhash_share_t* shares = nullptr;
    uint32_t* count = nullptr;
    uint32_t capacity = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

CfBuffers& cf_buffers()
{
    static CfBuffers b;
    return b;
}

int cf_reserve(CfBuffers& b, uint32_t max_shares)
{
    if (!b.start) {
        CUDA_CHECK(cudaEventCreate(&b.start));
        CUDA_CHECK(cudaEventCreate(&b.stop));
    }
    if (!b.count)
        CUDA_CHECK(cudaMalloc(&b.count, sizeof(uint32_t)));
    if (b.capacity >= max_shares && b.shares)
        return 0;
    if (b.shares)
        cudaFree(b.shares);
    b.shares = nullptr;
    b.capacity = 0;
    CUDA_CHECK(cudaMalloc(&b.shares, size_t(max_shares) * sizeof(qhash_share_t)));
    b.capacity = max_shares;
    return 0;
}

std::atomic<uint64_t> g_reverify_checked{0};
std::atomic<uint64_t> g_reverify_rejected{0};

/**
 * Phase 6.12: re-run every candidate through the statevector oracle on the CPU
 * before it can leave this function.
 *
 * The closed form agrees with the statevector on every one of the millions of
 * nonces the Phase-6 harness checked, but the two differ by ~1e-14 in ⟨Z⟩, and a
 * value sitting exactly on a Q1.15 rounding boundary has no defined FP64 answer.
 * Re-verifying closes that gap completely: the digest that gets submitted is the
 * oracle's, and it is re-tested against the target, so a boundary case can only
 * ever cost us a share, never produce an invalid one.
 *
 * Candidates are rare at any real difficulty, so this is free. It is skipped when
 * check_target is 0, because then every nonce is a "candidate" and the caller is a
 * test harness comparing raw digests, not a miner.
 */
uint32_t cf_reverify_candidates(const qhash_job_t* job, qhash_share_t* shares, uint32_t count)
{
    uint32_t kept = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t nonce = shares[i].nonce;
        uint8_t header[QHASH_INPUT_SIZE];
        std::memcpy(header, job->header, QHASH_INPUT_SIZE);
        header[76] = uint8_t(nonce);
        header[77] = uint8_t(nonce >> 8);
        header[78] = uint8_t(nonce >> 16);
        header[79] = uint8_t(nonce >> 24);

        uint8_t ref[QHASH_SHA256_SIZE];
        qhash_hash_cpu_sim(header, ref, QHASH_PRECISION_FP64, job->nTime,
                           QHASH_SIM_STATEVECTOR);
        ++g_reverify_checked;

        if (!qhash_hash_le_target(ref, job->target)) {
            ++g_reverify_rejected;
            continue;
        }
        shares[kept].nonce = nonce;
        std::memcpy(shares[kept].hash, ref, QHASH_SHA256_SIZE);
        ++kept;
    }
    return kept;
}

} // namespace

extern "C" void qhash_cuda_reverify_stats(uint64_t* checked, uint64_t* rejected)
{
    if (checked)
        *checked = g_reverify_checked.load();
    if (rejected)
        *rejected = g_reverify_rejected.load();
}

static int mine_batch_closed_form(const qhash_job_t* job, qhash_share_t* shares_out,
                                  uint32_t max_shares, uint32_t* share_count, double* seconds_out)
{
    const uint32_t n = job->nonce_count;
    if (n == 0)
        return 0;

    CfParams p{};
    sha256_midstate(job->header, p.midstate);
    p.hdr_w[0] = sha256_load_be(job->header + 64);
    p.hdr_w[1] = sha256_load_be(job->header + 68);
    p.hdr_w[2] = sha256_load_be(job->header + 72);
    sha256_pad_schedule(p.pad_w, QHASH_SHA256_SIZE + QHASH_NUM_QUBITS * sizeof(int16_t));
    /* Little-endian limbs, matching qhash_digest_le_target and cpuminer's layout. */
    for (int i = 0; i < 8; ++i)
        p.target[i] = uint32_t(job->target[4 * i]) | (uint32_t(job->target[4 * i + 1]) << 8) |
                      (uint32_t(job->target[4 * i + 2]) << 16) |
                      (uint32_t(job->target[4 * i + 3]) << 24);
    p.lut = host_angle_lut(angle_offset(job->nTime));
    p.nTime = job->nTime;
    p.check_target = job->check_target;

    int threads = job->threads_per_block > 0 ? job->threads_per_block : QHASH_CF_DEFAULT_THREADS;
    if (threads < 32)
        threads = 32;
    if (threads > 1024)
        threads = 1024;

    /* Persistent grid: fill every SM to its occupancy limit, then keep that many
       blocks resident and let the grid-stride loop walk the batch. Asking the
       occupancy API means the waves stay full whatever the register count ends up
       being after a compiler change. */
    int blocks = job->blocks;
    if (blocks <= 0) {
        const int sms = qhash_cuda_sm_count();
        int per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, qhash_cf_kernel, threads, 0) !=
                cudaSuccess ||
            per_sm < 1)
            per_sm = QHASH_CF_BLOCKS_PER_SM;
        blocks = (sms > 0 ? sms : 16) * per_sm * QHASH_CF_WAVES;
    }
    const int need = int((size_t(n) + size_t(threads) - 1) / size_t(threads));
    if (blocks > need)
        blocks = need;
    if (blocks < 1)
        blocks = 1;

    CfBuffers& buf = cf_buffers();
    std::lock_guard<std::mutex> lock(buf.mu);
    if (cf_reserve(buf, max_shares ? max_shares : 1) != 0)
        return -1;

    CUDA_CHECK(cudaMemcpyToSymbol(c_cf, &p, sizeof(p)));
    CUDA_CHECK(cudaMemset(buf.count, 0, sizeof(uint32_t)));

    CUDA_CHECK(cudaEventRecord(buf.start));

    qhash_cf_kernel<<<blocks, threads>>>(job->nonce_start, n, buf.shares, buf.count, max_shares);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(buf.stop));
    CUDA_CHECK(cudaEventSynchronize(buf.stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, buf.start, buf.stop));
    if (seconds_out)
        *seconds_out = double(ms) * 1e-3;

    uint32_t count = 0;
    CUDA_CHECK(cudaMemcpy(&count, buf.count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (count > max_shares)
        count = max_shares;
    if (count && shares_out) {
        CUDA_CHECK(cudaMemcpy(shares_out, buf.shares, count * sizeof(qhash_share_t),
                              cudaMemcpyDeviceToHost));
        if (job->check_target)
            count = cf_reverify_candidates(job, shares_out, count);
    }
    if (share_count)
        *share_count = count;
    return 0;
}

extern "C" int qhash_hash_gpu(const uint8_t header[QHASH_INPUT_SIZE],
                              uint8_t out[QHASH_SHA256_SIZE],
                              qhash_precision_t precision,
                              uint32_t nTime)
{
    qhash_job_t job{};
    std::memcpy(job.header, header, QHASH_INPUT_SIZE);
    job.nonce_start = uint32_t(header[76]) | (uint32_t(header[77]) << 8) |
                      (uint32_t(header[78]) << 16) | (uint32_t(header[79]) << 24);
    job.nonce_count = 1;
    job.nTime = nTime ? nTime
                      : (uint32_t(header[68]) | (uint32_t(header[69]) << 8) |
                         (uint32_t(header[70]) << 16) | (uint32_t(header[71]) << 24));
    job.precision = precision;
    job.check_target = 0;
    std::memset(job.target, 0xFF, 32);

    qhash_share_t share;
    uint32_t count = 0;
    double secs = 0;
    const int rc = qhash_mine_batch(&job, &share, 1, &count, &secs);
    if (rc != 0 || count != 1)
        return -1;
    std::memcpy(out, share.hash, 32);
    return 0;
}

template <typename Real>
static int mine_batch_typed(const qhash_job_t* job, qhash_share_t* shares_out, uint32_t max_shares,
                            uint32_t* share_count, double* seconds_out)
{
    const uint32_t n = job->nonce_count;
    if (n == 0)
        return 0;

    int threads = job->threads_per_block > 0 ? job->threads_per_block : QHASH_DEFAULT_THREADS;
    if (threads != 128 && threads != 256)
        threads = QHASH_DEFAULT_THREADS;

    int nstreams = job->num_streams > 0 ? job->num_streams : QHASH_DEFAULT_STREAMS;
    if (nstreams < 1)
        nstreams = 1;
    if (nstreams > QHASH_MAX_STREAMS)
        nstreams = QHASH_MAX_STREAMS;
    /* Multi-stream only helps when the batch can be split into concurrent slices. */
    if (nstreams > int(n))
        nstreams = int(n);

    Complex<Real>* d_states = nullptr;
    qhash_share_t* d_shares = nullptr;
    uint32_t* d_count = nullptr;
    uint8_t* d_header = nullptr;
    uint8_t* d_target = nullptr;

    const size_t state_bytes = size_t(n) * QHASH_STATE_SIZE * sizeof(Complex<Real>);
    CUDA_CHECK(cudaMalloc(&d_states, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_shares, max_shares * sizeof(qhash_share_t)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_header, QHASH_INPUT_SIZE));
    CUDA_CHECK(cudaMalloc(&d_target, QHASH_SHA256_SIZE));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_header, job->header, QHASH_INPUT_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, job->target, QHASH_SHA256_SIZE, cudaMemcpyHostToDevice));

    const int offset = angle_offset(job->nTime);

    cudaStream_t streams[QHASH_MAX_STREAMS];
    for (int s = 0; s < nstreams; ++s)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    /* Split the nonce range across streams so multiple grids overlap on the GPU. */
    const uint32_t base = (n + uint32_t(nstreams) - 1u) / uint32_t(nstreams);
    uint32_t cursor = 0;
    for (int s = 0; s < nstreams; ++s) {
        if (cursor >= n)
            break;
        const uint32_t slice = (cursor + base > n) ? (n - cursor) : base;
        qhash_mine_kernel<Real><<<int(slice), threads, 0, streams[s]>>>(
            d_header, job->nonce_start + cursor, job->nTime, offset, d_target, job->check_target,
            d_states + size_t(cursor) * QHASH_STATE_SIZE, d_shares, d_count, max_shares, nullptr);
        CUDA_CHECK(cudaGetLastError());
        cursor += slice;
    }

    for (int s = 0; s < nstreams; ++s)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (seconds_out)
        *seconds_out = double(ms) * 1e-3;

    uint32_t count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (count > max_shares)
        count = max_shares;
    if (share_count)
        *share_count = count;
    if (count && shares_out)
        CUDA_CHECK(cudaMemcpy(shares_out, d_shares, count * sizeof(qhash_share_t),
                              cudaMemcpyDeviceToHost));

    for (int s = 0; s < nstreams; ++s)
        cudaStreamDestroy(streams[s]);
    cudaFree(d_states);
    cudaFree(d_shares);
    cudaFree(d_count);
    cudaFree(d_header);
    cudaFree(d_target);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

extern "C" int qhash_mine_batch(const qhash_job_t* job, qhash_share_t* shares_out,
                                uint32_t max_shares, uint32_t* share_count, double* seconds_out)
{
    if (!qhash_cuda_available()) {
        /* CPU fallback */
        const auto t0 = std::chrono::steady_clock::now();
        uint32_t found = 0;
        for (uint32_t i = 0; i < job->nonce_count; ++i) {
            uint8_t header[QHASH_INPUT_SIZE];
            std::memcpy(header, job->header, QHASH_INPUT_SIZE);
            const uint32_t nonce = job->nonce_start + i;
            header[76] = uint8_t(nonce);
            header[77] = uint8_t(nonce >> 8);
            header[78] = uint8_t(nonce >> 16);
            header[79] = uint8_t(nonce >> 24);
            uint8_t hash[32];
            qhash_hash_cpu(header, hash, job->precision, job->nTime);
            const int ok = job->check_target ? qhash_hash_le_target(hash, job->target) : 1;
            if (ok && found < max_shares) {
                shares_out[found].nonce = nonce;
                std::memcpy(shares_out[found].hash, hash, 32);
                ++found;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (seconds_out)
            *seconds_out = std::chrono::duration<double>(t1 - t0).count();
        if (share_count)
            *share_count = found;
        return 0;
    }

    /* FP32 exists only to reproduce the legacy miner's statevector rounding, so it
       always takes the oracle path; the closed form is an FP64 identity. */
    if (job->precision == QHASH_PRECISION_FP32)
        return mine_batch_typed<float>(job, shares_out, max_shares, share_count, seconds_out);
    if (job->sim == QHASH_SIM_STATEVECTOR)
        return mine_batch_typed<double>(job, shares_out, max_shares, share_count, seconds_out);
    return mine_batch_closed_form(job, shares_out, max_shares, share_count, seconds_out);
}

#else /* QHASH_CPU_ONLY */

extern "C" int qhash_cuda_available(void) { return 0; }

extern "C" int qhash_cuda_sm_count(void) { return 0; }

extern "C" void qhash_cuda_reverify_stats(uint64_t* checked, uint64_t* rejected)
{
    if (checked)
        *checked = 0;
    if (rejected)
        *rejected = 0;
}

extern "C" int qhash_hash_gpu(const uint8_t header[QHASH_INPUT_SIZE],
                              uint8_t out[QHASH_SHA256_SIZE],
                              qhash_precision_t precision,
                              uint32_t nTime)
{
    qhash_hash_cpu(header, out, precision, nTime);
    return 0;
}

extern "C" int qhash_mine_batch(const qhash_job_t* job, qhash_share_t* shares_out,
                                uint32_t max_shares, uint32_t* share_count, double* seconds_out)
{
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t found = 0;
    for (uint32_t i = 0; i < job->nonce_count; ++i) {
        uint8_t header[QHASH_INPUT_SIZE];
        std::memcpy(header, job->header, QHASH_INPUT_SIZE);
        const uint32_t nonce = job->nonce_start + i;
        header[76] = uint8_t(nonce);
        header[77] = uint8_t(nonce >> 8);
        header[78] = uint8_t(nonce >> 16);
        header[79] = uint8_t(nonce >> 24);
        uint8_t hash[32];
        qhash_hash_cpu(header, hash, job->precision, job->nTime);
        const int ok = job->check_target ? qhash_hash_le_target(hash, job->target) : 1;
        if (ok && found < max_shares) {
            shares_out[found].nonce = nonce;
            std::memcpy(shares_out[found].hash, hash, 32);
            ++found;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (seconds_out)
        *seconds_out = std::chrono::duration<double>(t1 - t0).count();
    if (share_count)
        *share_count = found;
    return 0;
}

#endif /* QHASH_CPU_ONLY */
