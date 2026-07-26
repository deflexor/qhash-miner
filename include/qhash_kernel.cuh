#pragma once
/**
 * CUDA mining kernel API.
 *
 * Default path (Phase 6): the closed-form ⟨Z⟩ sweep, one nonce per thread, in a
 * persistent grid-stride kernel with no state buffer and no per-nonce barriers.
 *
 * Setting job.sim to QHASH_SIM_STATEVECTOR selects the original one-block-per-nonce
 * 65536-amplitude kernel, which is retained as the consensus oracle for A/B checks
 * and for re-verifying candidate nonces before a share is submitted.
 */
#include "qhash_cpu.h"
#include "qhash_params.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t nonce_start;
    uint32_t nonce_count;
    uint32_t nTime;                 /* soft-fork clock */
    qhash_precision_t precision;
    uint8_t header[QHASH_INPUT_SIZE]; /* template; nonce at bytes 76..79 overwritten */
    /* Little-endian 256-bit target, i.e. byte 31 most significant — the Bitcoin /
       cpuminer `work->target` layout, so a plain memcpy from it is correct.
       All-FF accepts everything, all-00 accepts nothing. See target.cuh. */
    uint8_t target[QHASH_SHA256_SIZE];
    int check_target;               /* 0 = record every digest (small batches only) */
    int threads_per_block;          /* 0 = default */
    int num_streams;                /* statevector path only: 1..4 chunk streams */
    /* Fields below were appended after the official-miner integration; that tree
       zero-initialises qhash_job_t, so 0 must always mean "default". */
    qhash_sim_t sim;                /* 0 = closed form, 1 = statevector oracle */
    int blocks;                     /* closed-form grid size; 0 = occupancy-derived */
} qhash_job_t;

/**
 * Launch geometry defaults.
 *
 * The statevector kernel wants 256 threads (one block per nonce, cooperating over
 * a 1 MiB state). The closed-form kernel is one nonce per thread and is retuned
 * from scratch in Phase 6.11 — the old numbers do not transfer.
 */
enum {
    QHASH_DEFAULT_THREADS = 256,
    QHASH_MAX_THREADS = 256,
    QHASH_DEFAULT_STREAMS = 1,
    QHASH_MAX_STREAMS = 4,
    /* Phase 6.11 sweep on an RTX 5060 Laptop: 128/256/512 threads land within 1%
       of each other (285.4 / 286.8 / 284.4 Mh/s), and resident waves from 1 to 16
       within 1% as well. The kernel is FP64-issue-bound, so occupancy stopped
       being the lever — these are just the middle of a flat region. */
    QHASH_CF_DEFAULT_THREADS = 256,
    /* Fallback only, for when the occupancy API cannot be queried. */
    QHASH_CF_BLOCKS_PER_SM = 8,
    /* Resident waves of full-occupancy blocks; more than one only helps to even
       out the tail when nonce_count is not a clean multiple of the grid. */
    QHASH_CF_WAVES = 4
};

typedef struct {
    uint32_t nonce;
    uint8_t hash[QHASH_SHA256_SIZE];
} qhash_share_t;

/**
 * Run batched hashes on GPU (or CPU fallback if built without CUDA).
 * @param shares_out  capacity = max_shares
 * @param share_count number written
 */
int qhash_mine_batch(const qhash_job_t* job,
                     qhash_share_t* shares_out,
                     uint32_t max_shares,
                     uint32_t* share_count,
                     double* seconds_out);

/** Hash a single header on the preferred backend. */
int qhash_hash_gpu(const uint8_t header[QHASH_INPUT_SIZE],
                   uint8_t out[QHASH_SHA256_SIZE],
                   qhash_precision_t precision,
                   uint32_t nTime);

int qhash_cuda_available(void);

/** Number of SMs on the current device, or 0 if there is none. */
int qhash_cuda_sm_count(void);

/**
 * Phase 6.12 counters. Every candidate the closed-form kernel finds is re-hashed
 * with the statevector oracle before it is returned as a share; `rejected` counts
 * candidates the oracle did not confirm against the target. It should stay 0 —
 * a nonzero value means a Q1.15 boundary case was caught doing exactly its job.
 */
void qhash_cuda_reverify_stats(uint64_t* checked, uint64_t* rejected);

#ifdef __cplusplus
}
#endif
