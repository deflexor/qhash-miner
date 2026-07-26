#pragma once
/**
 * Target comparison, in the one convention that matters for consensus.
 *
 * Bitcoin — and therefore Qubitcoin's CheckProofOfWork, and cpuminer's
 * valid_hash — reads a 256-bit hash **little-endian**: byte 0 of the SHA-256
 * output is the LEAST significant byte, and byte 31 the most significant. A
 * target is stored the same way, which is exactly the memory layout of
 * cpuminer's `work->target` (uint32_t[8], word 7 most significant, each word
 * little-endian on x86), so a plain memcpy into qhash_job_t::target is correct.
 *
 * Getting this backwards does not merely mis-rank hashes, it makes the filter
 * useless in one direction: comparing the digest's most significant byte
 * against the target's least significant byte accepts essentially everything.
 */
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define QHASH_TGT_HD __host__ __device__ __forceinline__
#else
#define QHASH_TGT_HD inline
#endif

/** True when `hash` <= `target`, both little-endian 32-byte integers. */
QHASH_TGT_HD bool qhash_hash_le_target(const uint8_t hash[32], const uint8_t target[32])
{
    for (int i = 31; i >= 0; --i) {
        if (hash[i] < target[i])
            return true;
        if (hash[i] > target[i])
            return false;
    }
    return true; /* equal meets the target */
}

/**
 * Word form for the closed-form kernel, which holds the digest as eight
 * big-endian SHA-256 words. Word k covers bytes 4k..4k+3, so byte-reversing it
 * gives the little-endian value of that 32-bit limb, and limb 7 is the most
 * significant. `target_le` must be the target's limbs in the same order.
 */
QHASH_TGT_HD bool qhash_digest_le_target(const uint32_t digest_be[8], const uint32_t target_le[8])
{
#pragma unroll
    for (int k = 7; k >= 0; --k) {
        const uint32_t d = digest_be[k];
        const uint32_t h = ((d & 0x000000FFu) << 24) | ((d & 0x0000FF00u) << 8) |
                           ((d & 0x00FF0000u) >> 8) | ((d & 0xFF000000u) >> 24);
        if (h < target_le[k])
            return true;
        if (h > target_le[k])
            return false;
    }
    return true;
}
