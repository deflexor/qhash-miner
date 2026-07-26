#pragma once
#include "qhash_params.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Full qhash: SHA256(header) → circuit → ⟨Z⟩ fixed → SHA256(hash||fixed).
 * @param header  80-byte block header (nTime at bytes 68..71 little-endian)
 * @param out     32-byte digest
 * @param precision QHASH_PRECISION_FP64 (node) or FP32 (legacy miner)
 * @param force_nTime if non-zero, override header nTime for soft-fork decisions
 */
void qhash_hash_cpu(const uint8_t header[QHASH_INPUT_SIZE],
                    uint8_t out[QHASH_SHA256_SIZE],
                    qhash_precision_t precision,
                    uint32_t force_nTime);

/** Circuit-only: nibbles in → Z expectations out (double). */
void qhash_simulate_cpu(const uint8_t nibbles[QHASH_NIBBLE_COUNT],
                        double expectations[QHASH_NUM_QUBITS],
                        qhash_precision_t precision,
                        uint32_t nTime);

#ifdef __cplusplus
}
#endif
