#pragma once
/**
 * Hardcoded 16-qubit / 2-layer qhash circuit topology.
 *
 * Per layer l ∈ {0,1}:
 *   for i ∈ [0,15]:  Ry(θ[2*l*16+i]), then Rz(θ[(2*l+1)*16+i])
 *   for i ∈ [0,14]:  CNOT(i → i+1)
 *
 * Angles (node consensus, nTime ≥ QHASH_SF_ANGLE):
 *   θ = −(2·nibble + 1) · π / 32
 * Legacy miner (pre-softfork / official miner):
 *   θ = −nibble · π / 16   ≡  −(2·nibble) · π / 32
 */
#include "gates.cuh"
#include "qhash_params.h"

namespace qhash {

QHASH_HD int angle_offset(uint32_t nTime)
{
    return (nTime >= QHASH_SF_ANGLE) ? 1 : 0;
}

/**
 * Rotation angle from a 4-bit nibble, matching qubitcoin-node:
 *   −(2·nibble + offset) · π / 32
 */
template <typename Real>
QHASH_HD Real nibble_angle(uint8_t nibble, int offset)
{
    return -Real(2 * int(nibble & 0xF) + offset) * Real(QHASH_PI) / Real(32);
}

QHASH_HD void split_nibbles(const uint8_t hash[QHASH_SHA256_SIZE], uint8_t nibbles[QHASH_NIBBLE_COUNT])
{
    for (int i = 0; i < QHASH_SHA256_SIZE; ++i) {
        nibbles[2 * i] = (hash[i] >> 4) & 0xF;
        nibbles[2 * i + 1] = hash[i] & 0xF;
    }
}

/**
 * Run the full qhash circuit on psi (must start as |0⟩).
 * nibbles: 64 values in 0..15 from SHA256(header).
 *
 * Default: separate Ry then Rz (matches cuStateVec ApplyPauliRotation order).
 * Gate matrices follow cuStateVec 1.14 empirically (exp(+i θ P), not θ/2).
 * Define QHASH_FUSE_RZ_RY for the Phase-2 fused 2×2 path.
 */
template <typename Real>
QHASH_HD void run_circuit(Complex<Real>* psi, const uint8_t nibbles[QHASH_NIBBLE_COUNT], int offset)
{
    for (int l = 0; l < QHASH_NUM_LAYERS; ++l) {
        for (int i = 0; i < QHASH_NUM_QUBITS; ++i) {
            const uint8_t ny = nibbles[(2 * l * QHASH_NUM_QUBITS + i) % QHASH_NIBBLE_COUNT];
            const uint8_t nz = nibbles[((2 * l + 1) * QHASH_NUM_QUBITS + i) % QHASH_NIBBLE_COUNT];
            const Real ty = nibble_angle<Real>(ny, offset);
            const Real tz = nibble_angle<Real>(nz, offset);
#ifdef QHASH_FUSE_RZ_RY
            const U2<Real> u = make_rz_ry<Real>(ty, tz);
            apply_u2(psi, i, u);
#else
            apply_u2(psi, i, make_ry<Real>(ty));
            apply_u2(psi, i, make_rz<Real>(tz));
#endif
        }
        for (int i = 0; i < QHASH_NUM_QUBITS - 1; ++i)
            apply_cnot(psi, i, i + 1);
    }
}

template <typename Real>
QHASH_HD void init_zero_state(Complex<Real>* psi)
{
    for (int i = 0; i < QHASH_STATE_SIZE; ++i)
        psi[i] = Complex<Real>(0, 0);
    psi[0] = Complex<Real>(1, 0);
}

template <typename Real>
QHASH_HD void measure_z_expectations(const Complex<Real>* psi, Real expectations[QHASH_NUM_QUBITS])
{
    for (int q = 0; q < QHASH_NUM_QUBITS; ++q)
        expectations[q] = expectation_z(psi, q);
}

/** Round to Q1.15 fixed-point (matches official miner toFixed / fpm). */
QHASH_HD int16_t to_fixed_q15(double x)
{
    const double mult = double(1 << QHASH_FRACTION_BITS);
    const int32_t y = (x >= 0.0) ? int32_t(x * mult + 0.5) : int32_t(x * mult - 0.5);
    /* Intentional int16 wrap for ±1.0 edge (32768 → -32768), matching miner cast. */
    return int16_t(y);
}

/**
 * Soft-fork zero-byte rejection (node Finalize).
 * Returns true if hash should be forced to all 0xFF.
 */
QHASH_HD bool softfork_reject_zeroes(uint32_t nTime, int zero_bytes)
{
    const int total = QHASH_NUM_QUBITS * int(sizeof(int16_t)); /* 32 */
    if (nTime >= QHASH_SF_ZERO_ALL && zero_bytes == total)
        return true;
    if (nTime >= QHASH_SF_ZERO_3_4 && zero_bytes >= total * 3 / 4)
        return true;
    if (nTime >= QHASH_SF_ZERO_1_4 && zero_bytes >= total / 4)
        return true;
    return false;
}

} // namespace qhash
