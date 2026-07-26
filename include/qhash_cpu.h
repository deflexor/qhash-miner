#pragma once
#include "qhash_params.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Which ⟨Z⟩ simulator to use.
 *
 * CLOSED_FORM is the Phase-6 16-step 2×2 sweep and is the production path.
 * STATEVECTOR is the original 65536-amplitude simulation, kept as the consensus
 * oracle for self-tests, the cuStateVec golden harness, and re-verification of
 * candidate nonces before a share is submitted.
 */
typedef enum {
    QHASH_SIM_CLOSED_FORM = 0,
    QHASH_SIM_STATEVECTOR = 1
} qhash_sim_t;

#define QHASH_DEFAULT_SIM QHASH_SIM_CLOSED_FORM

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

/** As qhash_hash_cpu, with an explicit simulator choice. */
void qhash_hash_cpu_sim(const uint8_t header[QHASH_INPUT_SIZE],
                        uint8_t out[QHASH_SHA256_SIZE],
                        qhash_precision_t precision,
                        uint32_t force_nTime,
                        qhash_sim_t sim);

/** Circuit-only: nibbles in → Z expectations out (double). */
void qhash_simulate_cpu(const uint8_t nibbles[QHASH_NIBBLE_COUNT],
                        double expectations[QHASH_NUM_QUBITS],
                        qhash_precision_t precision,
                        uint32_t nTime);

/** As qhash_simulate_cpu, with an explicit simulator choice. */
void qhash_simulate_cpu_sim(const uint8_t nibbles[QHASH_NIBBLE_COUNT],
                            double expectations[QHASH_NUM_QUBITS],
                            qhash_precision_t precision,
                            uint32_t nTime,
                            qhash_sim_t sim);

/**
 * Shared tail of the hash: ⟨Z⟩ → Q1.15 little-endian → soft-fork zero-byte
 * rejection → SHA256(header_hash ‖ fixed). Exposed so tests and the GPU path
 * can be compared against exactly the same finalize logic.
 */
void qhash_finalize_cpu(const uint8_t in_hash[QHASH_SHA256_SIZE],
                        const double expectations[QHASH_NUM_QUBITS],
                        uint32_t nTime,
                        uint8_t out[QHASH_SHA256_SIZE]);

#ifdef __cplusplus
}
#endif
