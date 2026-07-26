#pragma once
/**
 * Phase-1 monolithic CUDA mining kernel API.
 * One block per nonce; threads cooperate on a shared global state vector.
 */
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
    uint8_t target[QHASH_SHA256_SIZE]; /* little-endian 256-bit target; all-FF = no check */
    int check_target;               /* 0 = hash only / benchmark */
    int threads_per_block;          /* 128 or 256; 0 = default (256) */
    int num_streams;                /* 1..4 concurrent chunk streams; 0 = default (1) */
} qhash_job_t;

/** Default launch geometry (Phase 3 occupancy sweep). */
enum {
    QHASH_DEFAULT_THREADS = 256,
    QHASH_MAX_THREADS = 256,
    QHASH_DEFAULT_STREAMS = 1,
    QHASH_MAX_STREAMS = 4
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

#ifdef __cplusplus
}
#endif
