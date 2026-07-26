#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Canonical qhash parameters from super-quantum/qubitcoin(+miner). */
enum {
    QHASH_NUM_QUBITS = 16,
    QHASH_NUM_LAYERS = 2,
    QHASH_STATE_SIZE = 1 << QHASH_NUM_QUBITS, /* 65536 */
    QHASH_SHA256_SIZE = 32,
    QHASH_INPUT_SIZE = 80, /* block header */
    QHASH_NIBBLE_COUNT = 64, /* 2 * SHA256_SIZE */
    QHASH_FRACTION_BITS = 15
};

/**
 * Soft-fork activation times from qubitcoin-node/src/crypto/qhash.cpp
 * Angle offset activates at SF_ANGLE; zero-byte rejection thresholds earlier.
 */
enum {
    QHASH_SF_ZERO_ALL = 1753105444u,   /* Jul 21 2025: all-zero fixed bytes → 0xFF.. hash */
    QHASH_SF_ZERO_3_4 = 1753305380u,   /* Jul 24 2025: ≥75% zero bytes → 0xFF.. */
    QHASH_SF_ZERO_1_4 = 1754220531u,   /* Aug  3 2025: ≥25% zero bytes → 0xFF.. */
    QHASH_SF_ANGLE    = 1758762000u    /* Sep 25 2025: +(1)*π/32 angle offset */
};

/**
 * Precision mode:
 *  - NODE  (FP64): matches consensus node (CUDA_C_64F)
 *  - MINER (FP32): matches official qubitcoin-miner (CUDA_C_32F)
 */
typedef enum {
    QHASH_PRECISION_FP64 = 0,
    QHASH_PRECISION_FP32 = 1
} qhash_precision_t;

#ifdef __cplusplus
}
#endif
