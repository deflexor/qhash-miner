/**
 * Monolithic fused qhash CUDA kernel (Phase 1–3b+).
 *
 * Launch geometry: gridDim.x = number of nonces, blockDim.x = 128 or 256.
 * Each block owns one state vector in global memory (L2-resident working set).
 * Ry+Rz fused into one 2×2 per qubit/layer; high U2 fibers first (U2s commute),
 * then low U2+CNOT fused in one tile residency; optional dual-tile so boundary
 * CNOT(kTileQ-1,kTileQ) stays in smem; high CNOT fibers; ⟨Z⟩ + SHA256 inline.
 * Optional multi-stream; optional 2× tile dbuf (usually off).
 *
 * When compiled without CUDA (QHASH_CPU_ONLY), falls back to the CPU simulator.
 */
#include "qhash_kernel.cuh"
#include "circuit.cuh"
#include "sha256.cuh"
#include "qhash_cpu.h"

#include <chrono>
#include <cstdio>
#include <cstring>
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

/*
 * Shared-memory tile: default 2048 amps = 2^11 → gates on qubits 0..10 stay
 * inside a tile (FP64 ≈ 32 KiB + gate cache). Single-qubit U2s in a layer
 * commute, so low-qubit gates may run in the tile before high-qubit global U2s.
 *
 * Tile 4096 (qubits 0..11) needs dynamic smem for FP64 (65 KiB > 48 KiB static);
 * enable with -DQHASH_SMEM_TILE=4096 (host opts in via cudaFuncSetAttribute).
 * Override with -DQHASH_SMEM_TILE=0 to disable (CNOT pair-index tiling remains).
 *
 * Note: even/odd CNOT brickwork is NOT consensus-safe here — NN CNOTs in a
 * chain do not commute; must stay sequential for bit-exact digests.
 */
#ifndef QHASH_SMEM_TILE
#define QHASH_SMEM_TILE 2048
#endif
#ifndef QHASH_SMEM_TILE_Q
#if QHASH_SMEM_TILE <= 0
#define QHASH_SMEM_TILE_Q 0
#elif QHASH_SMEM_TILE == 4096
#define QHASH_SMEM_TILE_Q 12
#elif QHASH_SMEM_TILE == 2048
#define QHASH_SMEM_TILE_Q 11
#elif QHASH_SMEM_TILE == 1024
#define QHASH_SMEM_TILE_Q 10
#else
#error "Set QHASH_SMEM_TILE to 0, 1024, 2048, or 4096 (and matching QHASH_SMEM_TILE_Q)"
#endif
#endif

/*
 * Optional 2× tile double-buffer (scatter N ∥ gather N+1). Measured slower on
 * RTX 5060 Laptop (extra dyn smem / pointer ping-pong) — leave OFF.
 * Fiber-blocked high qubits (gather→gates→scatter) stay ON and help.
 *
 * QHASH_BOUNDARY_PAIR=1: low section uses 2×tile smem so CNOT(kTileQ-1,kTileQ)
 * stays in shared memory (no full-state global boundary pass). Pays dyn smem.
 */
#ifndef QHASH_TILE_DBUF
#define QHASH_TILE_DBUF 0
#endif
#ifndef QHASH_BOUNDARY_PAIR
#define QHASH_BOUNDARY_PAIR 0
#endif
#if QHASH_TILE_DBUF && (QHASH_SMEM_TILE > 0) && (QHASH_SMEM_TILE * 16 * 2 <= 96 * 1024)
#define QHASH_DYN_TILES 2
#elif QHASH_BOUNDARY_PAIR && (QHASH_SMEM_TILE > 0) && (QHASH_SMEM_TILE * 16 * 2 <= 96 * 1024)
#define QHASH_DYN_TILES 2
#else
#define QHASH_DYN_TILES 1
#endif
#if (QHASH_SMEM_TILE * 16 > 48 * 1024) || (QHASH_DYN_TILES > 1)
#define QHASH_SMEM_DYNAMIC 1
#else
#define QHASH_SMEM_DYNAMIC 0
#endif

/**
 * High-qubit fiber block (Phase 3b+): qubits [kTileQ .. 15] are the high bits
 * above a contiguous smem tile. For fixed low index L (0..2^kTileQ-1), the
 * 2^(16-kTileQ) amplitudes {psi[L + j<<kTileQ]} form a fiber closed under all
 * gates on those high qubits. Gather fibers into the tile, apply every high
 * U2/CNOT in smem, scatter once — vs one full-state global pass per gate.
 *
 * Tile layout while fiber-blocked: tile[f * kFiber + j] holds fiber
 * (fiber_base + f), amplitude j. Bit-exact: fibers commute; within a fiber
 * gate order matches the sequential global path.
 *
 * Gather/scatter stream consecutive fibers at fixed j for coalesced global IO.
 */
template <typename Real>
__device__ void d_fiber_gather(Complex<Real>* tile, const Complex<Real>* psi, int fiber_base,
                               int n_fibers, int fiber_len, int low_bits, int tid, int nthreads)
{
    const int n = n_fibers * fiber_len;
    for (int i = tid; i < n; i += nthreads) {
        const int j = i / n_fibers;
        const int f = i - j * n_fibers;
        tile[f * fiber_len + j] = psi[(fiber_base + f) | (j << low_bits)];
    }
    __syncthreads();
}

template <typename Real>
__device__ void d_fiber_scatter(Complex<Real>* psi, const Complex<Real>* tile, int fiber_base,
                                int n_fibers, int fiber_len, int low_bits, int tid, int nthreads)
{
    const int n = n_fibers * fiber_len;
    for (int i = tid; i < n; i += nthreads) {
        const int j = i / n_fibers;
        const int f = i - j * n_fibers;
        psi[(fiber_base + f) | (j << low_bits)] = tile[f * fiber_len + j];
    }
    __syncthreads();
}

/** Scatter tile_a[fiber_base] and optionally gather tile_b[next_base] in one pass. */
template <typename Real>
__device__ void d_fiber_scatter_gather(Complex<Real>* psi, const Complex<Real>* tile_out,
                                       Complex<Real>* tile_in, int fiber_base, int next_base,
                                       int do_gather, int n_fibers, int fiber_len, int low_bits,
                                       int tid, int nthreads)
{
    const int n = n_fibers * fiber_len;
    for (int i = tid; i < n; i += nthreads) {
        const int j = i / n_fibers;
        const int f = i - j * n_fibers;
        psi[(fiber_base + f) | (j << low_bits)] = tile_out[f * fiber_len + j];
        if (do_gather)
            tile_in[f * fiber_len + j] = psi[(next_base + f) | (j << low_bits)];
    }
    __syncthreads();
}

/** U2 on relative qubit rq inside each packed fiber (n_fibers × fiber_len). */
template <typename Real>
__device__ void d_fiber_u2(Complex<Real>* tile, int rq, const U2<Real>& u, int n_fibers,
                           int fiber_len, int tid, int nthreads)
{
    const int npairs_fiber = fiber_len >> 1;
    const int total = n_fibers * npairs_fiber;
    const int mask = 1 << rq;
    for (int p = tid; p < total; p += nthreads) {
        const int f = p / npairs_fiber;
        const int pf = p - f * npairs_fiber;
        const int lo = pf & ((1 << rq) - 1);
        const int hi = pf >> rq;
        const int i = (f * fiber_len) + (lo | (hi << (rq + 1)));
        const int j = i | mask;
        Complex<Real> a = tile[i];
        Complex<Real> b = tile[j];
        apply_u2_pair(a, b, u);
        tile[i] = a;
        tile[j] = b;
    }
    __syncthreads();
}

/** CNOT on relative (rc → rt) inside each packed fiber. */
template <typename Real>
__device__ void d_fiber_cnot(Complex<Real>* tile, int rc, int rt, int n_fibers, int fiber_len,
                             int tid, int nthreads)
{
    const int npairs_fiber = fiber_len >> 2;
    const int total = n_fibers * npairs_fiber;
    const int cmask = 1 << rc;
    const int tmask = 1 << rt;
    const int q0 = rc < rt ? rc : rt;
    const int q1 = rc < rt ? rt : rc;
    const int lo_bits = q0;
    const int mid_bits = q1 - q0 - 1;
    const int lo_mask = (1 << lo_bits) - 1;
    const int mid_mask = (1 << mid_bits) - 1;

    for (int p = tid; p < total; p += nthreads) {
        const int f = p / npairs_fiber;
        const int pf = p - f * npairs_fiber;
        const int lo = pf & lo_mask;
        const int mid = (pf >> lo_bits) & mid_mask;
        const int hi = pf >> (lo_bits + mid_bits);
        const int base = (f * fiber_len) + (lo | (mid << (q0 + 1)) | (hi << (q1 + 1)));
        const int i = base | cmask;
        const int j = i | tmask;
        const Complex<Real> tmp = tile[i];
        tile[i] = tile[j];
        tile[j] = tmp;
    }
    __syncthreads();
}

template <typename Real>
__device__ void d_run_circuit(Complex<Real>* psi, const uint8_t* nibbles, int offset, int tid,
                              int nthreads, char* dyn_tile)
{
#if defined(QHASH_FUSE_RZ_RY) && (QHASH_SMEM_TILE > 0)
    constexpr int kTile = QHASH_SMEM_TILE;
    constexpr int kTileQ = QHASH_SMEM_TILE_Q;
    constexpr int kFiber = 1 << (QHASH_NUM_QUBITS - kTileQ);
    constexpr int kFibersPerTile = kTile / kFiber;
    constexpr int kNumFibers = 1 << kTileQ;
    /* Raw bytes avoid non-trivial Complex/U2 ctors in __shared__ (nvcc #20054). */
#if QHASH_SMEM_DYNAMIC
    Complex<Real>* tile = reinterpret_cast<Complex<Real>*>(dyn_tile);
#if QHASH_DYN_TILES > 1
    Complex<Real>* tile_b = tile + kTile;
#else
    Complex<Real>* tile_b = tile;
#endif
#else
    (void)dyn_tile;
    __shared__ __align__(16) char tile_raw[kTile * sizeof(Complex<Real>)];
    Complex<Real>* tile = reinterpret_cast<Complex<Real>*>(tile_raw);
    Complex<Real>* tile_b = tile;
#endif
    __shared__ __align__(16) char us_raw[QHASH_NUM_QUBITS * sizeof(U2<Real>)];
    U2<Real>* us = reinterpret_cast<U2<Real>*>(us_raw);

    for (int l = 0; l < QHASH_NUM_LAYERS; ++l) {
        if (tid < QHASH_NUM_QUBITS) {
            const uint8_t ny =
                nibbles[(2 * l * QHASH_NUM_QUBITS + tid) % QHASH_NIBBLE_COUNT];
            const uint8_t nz =
                nibbles[((2 * l + 1) * QHASH_NUM_QUBITS + tid) % QHASH_NIBBLE_COUNT];
            us[tid] = make_rz_ry<Real>(nibble_angle<Real>(ny, offset),
                                       nibble_angle<Real>(nz, offset));
        }
        __syncthreads();

#if QHASH_DYN_TILES > 1
        /* Low U2: double-buffered contiguous tiles. */
        {
            Complex<Real>* cur = tile;
            Complex<Real>* nxt = tile_b;
            for (int i = tid; i < kTile; i += nthreads)
                cur[i] = psi[i];
            __syncthreads();
            for (int base = 0; base < QHASH_STATE_SIZE; base += kTile) {
                for (int q = 0; q < kTileQ; ++q)
                    d_apply_u2_n(cur, q, us[q], tid, nthreads, kTile);
                const int next = base + kTile;
                const int has_next = next < QHASH_STATE_SIZE;
                for (int i = tid; i < kTile; i += nthreads) {
                    psi[base + i] = cur[i];
                    if (has_next)
                        nxt[i] = psi[next + i];
                }
                __syncthreads();
                Complex<Real>* tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
        }

        /* High U2: fiber double-buffer. */
        {
            Complex<Real>* cur = tile;
            Complex<Real>* nxt = tile_b;
            d_fiber_gather(cur, psi, 0, kFibersPerTile, kFiber, kTileQ, tid, nthreads);
            for (int fb = 0; fb < kNumFibers; fb += kFibersPerTile) {
                for (int q = kTileQ; q < QHASH_NUM_QUBITS; ++q)
                    d_fiber_u2(cur, q - kTileQ, us[q], kFibersPerTile, kFiber, tid, nthreads);
                const int next = fb + kFibersPerTile;
                d_fiber_scatter_gather(psi, cur, nxt, fb, next, next < kNumFibers, kFibersPerTile,
                                       kFiber, kTileQ, tid, nthreads);
                Complex<Real>* tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
        }

        /* Low CNOT: double-buffered tiles. */
        {
            Complex<Real>* cur = tile;
            Complex<Real>* nxt = tile_b;
            for (int i = tid; i < kTile; i += nthreads)
                cur[i] = psi[i];
            __syncthreads();
            for (int base = 0; base < QHASH_STATE_SIZE; base += kTile) {
                for (int c = 0; c < kTileQ - 1; ++c)
                    d_apply_cnot_n(cur, c, c + 1, tid, nthreads, kTile);
                const int next = base + kTile;
                const int has_next = next < QHASH_STATE_SIZE;
                for (int i = tid; i < kTile; i += nthreads) {
                    psi[base + i] = cur[i];
                    if (has_next)
                        nxt[i] = psi[next + i];
                }
                __syncthreads();
                Complex<Real>* tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
        }
        d_apply_cnot(psi, kTileQ - 1, kTileQ, tid, nthreads);

        /* High CNOT: fiber double-buffer. */
        {
            Complex<Real>* cur = tile;
            Complex<Real>* nxt = tile_b;
            d_fiber_gather(cur, psi, 0, kFibersPerTile, kFiber, kTileQ, tid, nthreads);
            for (int fb = 0; fb < kNumFibers; fb += kFibersPerTile) {
                for (int c = kTileQ; c < QHASH_NUM_QUBITS - 1; ++c)
                    d_fiber_cnot(cur, c - kTileQ, c + 1 - kTileQ, kFibersPerTile, kFiber, tid,
                                 nthreads);
                const int next = fb + kFibersPerTile;
                d_fiber_scatter_gather(psi, cur, nxt, fb, next, next < kNumFibers, kFibersPerTile,
                                       kFiber, kTileQ, tid, nthreads);
                Complex<Real>* tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
        }
#else
        /*
         * U2s on distinct qubits commute → do high U2s first, then fuse low U2
         * + low CNOT in one tile residency (halves contiguous gather/scatter).
         */
        for (int fb = 0; fb < kNumFibers; fb += kFibersPerTile) {
            d_fiber_gather(tile, psi, fb, kFibersPerTile, kFiber, kTileQ, tid, nthreads);
#pragma unroll
            for (int q = kTileQ; q < QHASH_NUM_QUBITS; ++q)
                d_fiber_u2(tile, q - kTileQ, us[q], kFibersPerTile, kFiber, tid, nthreads);
            d_fiber_scatter(psi, tile, fb, kFibersPerTile, kFiber, kTileQ, tid, nthreads);
        }

#if QHASH_BOUNDARY_PAIR && (QHASH_DYN_TILES > 1)
        /* Dual-tile low section: U2 0..kTileQ-1 + CNOT 0..kTileQ in 2×kTile smem. */
        {
            constexpr int kPair = kTile * 2;
            for (int base = 0; base < QHASH_STATE_SIZE; base += kPair) {
                for (int i = tid; i < kPair; i += nthreads)
                    tile[i] = psi[base + i];
                __syncthreads();
#pragma unroll
                for (int q = 0; q < kTileQ; ++q)
                    d_apply_u2_n(tile, q, us[q], tid, nthreads, kPair);
#pragma unroll
                for (int c = 0; c < kTileQ; ++c)
                    d_apply_cnot_n(tile, c, c + 1, tid, nthreads, kPair);
                for (int i = tid; i < kPair; i += nthreads)
                    psi[base + i] = tile[i];
                __syncthreads();
            }
        }
#else
        /* Single-tile: U2 + CNOT 0..kTileQ-2 fused; boundary CNOT stays global. */
        for (int base = 0; base < QHASH_STATE_SIZE; base += kTile) {
            for (int i = tid; i < kTile; i += nthreads)
                tile[i] = psi[base + i];
            __syncthreads();
#pragma unroll
            for (int q = 0; q < kTileQ; ++q)
                d_apply_u2_n(tile, q, us[q], tid, nthreads, kTile);
#pragma unroll
            for (int c = 0; c < kTileQ - 1; ++c)
                d_apply_cnot_n(tile, c, c + 1, tid, nthreads, kTile);
            for (int i = tid; i < kTile; i += nthreads)
                psi[base + i] = tile[i];
            __syncthreads();
        }
        d_apply_cnot(psi, kTileQ - 1, kTileQ, tid, nthreads);
#endif

        /* High NN CNOTs: fiber-blocked (controls ≥ kTileQ). */
        for (int fb = 0; fb < kNumFibers; fb += kFibersPerTile) {
            d_fiber_gather(tile, psi, fb, kFibersPerTile, kFiber, kTileQ, tid, nthreads);
#pragma unroll
            for (int c = kTileQ; c < QHASH_NUM_QUBITS - 1; ++c)
                d_fiber_cnot(tile, c - kTileQ, c + 1 - kTileQ, kFibersPerTile, kFiber, tid,
                             nthreads);
            d_fiber_scatter(psi, tile, fb, kFibersPerTile, kFiber, kTileQ, tid, nthreads);
        }
#endif
    }
#else
    (void)dyn_tile;
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
#endif
}

/**
 * ⟨Z⟩ for all qubits: one state scan, then per-qubit tree reduce.
 * Per-thread accumulation order for each qubit matches the old 16-pass loop
 * (i = tid, tid+nthreads, …); tree-reduce order unchanged — bit-exact Q1.15.
 */
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
    /* Compare as little-endian 256-bit integers (Bitcoin style: hash[::-1] < target).
       Mining typically compares byte-reversed; here we do raw memcmp big-endian of
       the SHA256 digest as produced (big-endian words). Match cpuminer: treat as
       32-byte BE number. */
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < target[i])
            return 1;
        if (hash[i] > target[i])
            return 0;
    }
    return 1;
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
#if QHASH_SMEM_DYNAMIC
    extern __shared__ __align__(16) char dyn_smem[];
    d_run_circuit(psi, s_nibbles, offset, tid, nthreads, dyn_smem);
#else
    d_run_circuit(psi, s_nibbles, offset, tid, nthreads, nullptr);
#endif
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

extern "C" int qhash_cuda_available(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
        return 0;
    return n > 0;
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

#if QHASH_SMEM_DYNAMIC
    /* Opt-in past 48 KiB static cap (Blackwell Laptop: sharedMemPerBlockOptin ≈ 99 KiB). */
    const size_t dyn_smem =
        size_t(QHASH_DYN_TILES) * size_t(QHASH_SMEM_TILE) * sizeof(Complex<Real>);
    CUDA_CHECK(cudaFuncSetAttribute(qhash_mine_kernel<Real>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, int(dyn_smem)));
#else
    const size_t dyn_smem = 0;
#endif

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
        qhash_mine_kernel<Real><<<int(slice), threads, dyn_smem, streams[s]>>>(
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
            int ok = 1;
            if (job->check_target) {
                ok = 1;
                for (int b = 0; b < 32; ++b) {
                    if (hash[b] < job->target[b]) {
                        ok = 1;
                        break;
                    }
                    if (hash[b] > job->target[b]) {
                        ok = 0;
                        break;
                    }
                }
            }
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

    if (job->precision == QHASH_PRECISION_FP32)
        return mine_batch_typed<float>(job, shares_out, max_shares, share_count, seconds_out);
    return mine_batch_typed<double>(job, shares_out, max_shares, share_count, seconds_out);
}

#else /* QHASH_CPU_ONLY */

extern "C" int qhash_cuda_available(void) { return 0; }

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
        int ok = 1;
        if (job->check_target) {
            for (int b = 0; b < 32; ++b) {
                if (hash[b] < job->target[b])
                    break;
                if (hash[b] > job->target[b]) {
                    ok = 0;
                    break;
                }
            }
        }
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
