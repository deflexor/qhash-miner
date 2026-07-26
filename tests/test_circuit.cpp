/**
 * Correctness & SHA-256 unit tests (CPU). Run without a GPU.
 */
#include "circuit.cuh"
#include "qhash_cpu.h"
#include "sha256.cuh"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

using namespace qhash;

static int g_fail = 0;

#define EXPECT(cond, msg)                                                                          \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL: %s\n", msg);                                               \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static void test_sha256_empty()
{
    uint8_t out[32];
    const uint8_t empty = 0;
    sha256(&empty, 0, out);
    static const uint8_t exp[32] = {
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
        0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
        0xb8, 0x55};
    EXPECT(std::memcmp(out, exp, 32) == 0, "SHA256(\"\") NIST vector");
}

static void test_sha256_abc()
{
    const char* msg = "abc";
    uint8_t out[32];
    sha256(reinterpret_cast<const uint8_t*>(msg), 3, out);
    static const uint8_t exp[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22,
        0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00,
        0x15, 0xad};
    EXPECT(std::memcmp(out, exp, 32) == 0, "SHA256(\"abc\") NIST vector");
}

static void test_identity_circuit_measure()
{
    /* |0⟩^⊗16 → all ⟨Z⟩ = +1 */
    std::vector<Complex<double>> psi(QHASH_STATE_SIZE);
    init_zero_state(psi.data());
    double e[QHASH_NUM_QUBITS];
    for (int q = 0; q < QHASH_NUM_QUBITS; ++q)
        e[q] = expectation_z(psi.data(), q);
    for (int q = 0; q < QHASH_NUM_QUBITS; ++q)
        EXPECT(std::fabs(e[q] - 1.0) < 1e-12, "|0> Z expectation == 1");
}

static void test_ry_pi()
{
    /* ApplyPauliRotation(π/2, Y) = exp(+i π/2 Y): |0⟩ → -|1⟩, ⟨Z₀⟩ = −1 */
    std::vector<Complex<double>> psi(QHASH_STATE_SIZE);
    init_zero_state(psi.data());
    const U2<double> u = make_rz_ry<double>(QHASH_PI / 2, 0.0);
    apply_u2(psi.data(), 0, u);
    const double z0 = expectation_z(psi.data(), 0);
    EXPECT(std::fabs(z0 + 1.0) < 1e-9, "Ry(pi/2)|0> => Z=-1 (cuStateVec convention)");
}

static void test_cnot_entangle()
{
    /* |+⟩⊗|0⟩ then CNOT → Bell; ⟨Z0⟩=0, ⟨Z1⟩=0
       |+⟩ via ApplyPauliRotation(-π/4, Y) under cuStateVec exp(+iθY) convention. */
    std::vector<Complex<double>> psi(QHASH_STATE_SIZE);
    init_zero_state(psi.data());
    const U2<double> to_plus = make_rz_ry<double>(-QHASH_PI / 4, 0.0);
    apply_u2(psi.data(), 0, to_plus);
    apply_cnot(psi.data(), 0, 1);
    const double z0 = expectation_z(psi.data(), 0);
    const double z1 = expectation_z(psi.data(), 1);
    EXPECT(std::fabs(z0) < 1e-9, "Bell Z0≈0");
    EXPECT(std::fabs(z1) < 1e-9, "Bell Z1≈0");
}

static void test_fixed_point()
{
    EXPECT(to_fixed_q15(0.0) == 0, "fixed 0");
    EXPECT(to_fixed_q15(0.5) == 16384, "fixed 0.5");
    /* ±1.0 → 32768 / -32768 intermediate; int16 wrap of +32768 is -32768 */
    EXPECT(to_fixed_q15(1.0) == int16_t(32768), "fixed 1.0 wraps");
    EXPECT(to_fixed_q15(-1.0) == int16_t(-32768), "fixed -1.0");
}

static void test_fp32_fp64_agreement()
{
    uint8_t header[QHASH_INPUT_SIZE];
    std::memset(header, 0x3C, sizeof(header));
    const uint32_t nTime = QHASH_SF_ANGLE;
    header[68] = uint8_t(nTime);
    header[69] = uint8_t(nTime >> 8);
    header[70] = uint8_t(nTime >> 16);
    header[71] = uint8_t(nTime >> 24);

    uint8_t nibbles[QHASH_NIBBLE_COUNT];
    uint8_t hash[32];
    sha256(header, QHASH_INPUT_SIZE, hash);
    split_nibbles(hash, nibbles);

    double e64[QHASH_NUM_QUBITS], e32[QHASH_NUM_QUBITS];
    qhash_simulate_cpu(nibbles, e64, QHASH_PRECISION_FP64, nTime);
    qhash_simulate_cpu(nibbles, e32, QHASH_PRECISION_FP32, nTime);

    int mismatch_fixed = 0;
    for (int i = 0; i < QHASH_NUM_QUBITS; ++i) {
        if (to_fixed_q15(e64[i]) != to_fixed_q15(e32[i]))
            ++mismatch_fixed;
    }
    std::printf("FP32 vs FP64 fixed-point mismatches on sample: %d / %d\n", mismatch_fixed,
                QHASH_NUM_QUBITS);
    /* Informational — do not fail; Phase 4 decides FP32 viability. */
}

int main()
{
    test_sha256_empty();
    test_sha256_abc();
    test_identity_circuit_measure();
    test_ry_pi();
    test_cnot_entangle();
    test_fixed_point();
    test_fp32_fp64_agreement();

    if (g_fail) {
        std::fprintf(stderr, "%d test(s) failed\n", g_fail);
        return 1;
    }
    std::printf("All tests passed\n");
    return 0;
}
