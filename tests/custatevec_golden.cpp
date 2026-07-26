/**
 * Golden harness: cuStateVec FP64 (the consensus node's simulator) vs our two
 * implementations — the FP64 statevector and the Phase-6 closed form.
 *
 * Both are scored against cuStateVec independently, which is what makes the
 * comparison meaningful: if the closed form and the statevector ever disagree, the
 * question is which one matches the node, not which one matches the other.
 *
 * Builds only when QHASH_ENABLE_CUSTATEVEC=ON and cuQuantum + cuBLAS 13 are found.
 *
 * Usage:
 *   qhash-custatevec-golden [--nonces N] [--ntime T] [--tol EPS] [--uniform]
 *
 * --uniform replaces the nonce scan with an exhaustive sweep of the degenerate
 * "same nibble on every qubit" patterns, which is where ⟨Z⟩ lands exactly on a
 * Q1.15 rounding boundary and no FP64 simulator has a well-defined answer.
 */
#include "circuit.cuh"
#include "closed_form.cuh"
#include "qhash_cpu.h"
#include "qhash_params.h"
#include "reference_ld.h"
#include "sha256.cuh"

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <custatevec.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CSV_CHECK(x)                                                                               \
    do {                                                                                           \
        const custatevecStatus_t e_ = (x);                                                         \
        if (e_ != CUSTATEVEC_STATUS_SUCCESS) {                                                     \
            std::fprintf(stderr, "cuStateVec error %s at %s:%d\n", custatevecGetErrorString(e_),   \
                         __FILE__, __LINE__);                                                      \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        const cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,     \
                         __LINE__);                                                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

namespace {

constexpr int kQ = QHASH_NUM_QUBITS;
constexpr int kLayers = QHASH_NUM_LAYERS;

static const cuDoubleComplex kMatrixX[] = {
    {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

struct CsvCtx {
    custatevecHandle_t handle{};
    cuDoubleComplex* d_state = nullptr;
    void* extra = nullptr;
    size_t extra_size = 0;
};

int csv_init(CsvCtx& ctx)
{
    CSV_CHECK(custatevecCreate(&ctx.handle));
    const size_t bytes = size_t(1u << kQ) * sizeof(cuDoubleComplex);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.d_state), bytes));
    CSV_CHECK(custatevecApplyMatrixGetWorkspaceSize(
        ctx.handle, CUDA_C_64F, kQ, kMatrixX, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 1,
        CUSTATEVEC_COMPUTE_DEFAULT, &ctx.extra_size));
    if (ctx.extra_size)
        CUDA_CHECK(cudaMalloc(&ctx.extra, ctx.extra_size));
    return 0;
}

void csv_destroy(CsvCtx& ctx)
{
    if (ctx.extra)
        cudaFree(ctx.extra);
    if (ctx.d_state)
        cudaFree(ctx.d_state);
    if (ctx.handle)
        custatevecDestroy(ctx.handle);
    ctx = {};
}

int csv_simulate(CsvCtx& ctx, const uint8_t nibbles[QHASH_NIBBLE_COUNT], uint32_t nTime,
                 double expectations[kQ])
{
    const int offset = (nTime >= QHASH_SF_ANGLE) ? 1 : 0;
    static const custatevecPauli_t pauliY[] = {CUSTATEVEC_PAULI_Y};
    static const custatevecPauli_t pauliZ[] = {CUSTATEVEC_PAULI_Z};

    CSV_CHECK(custatevecInitializeStateVector(ctx.handle, ctx.d_state, CUDA_C_64F, kQ,
                                              CUSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    for (int l = 0; l < kLayers; ++l) {
        for (int i = 0; i < kQ; ++i) {
            const int32_t target = i;
            const uint8_t ny = nibbles[(2 * l * kQ + i) % QHASH_NIBBLE_COUNT];
            const uint8_t nz = nibbles[((2 * l + 1) * kQ + i) % QHASH_NIBBLE_COUNT];
            const double ty = -(2 * int(ny) + offset) * M_PI / 32.0;
            const double tz = -(2 * int(nz) + offset) * M_PI / 32.0;
            CSV_CHECK(custatevecApplyPauliRotation(ctx.handle, ctx.d_state, CUDA_C_64F, kQ, ty,
                                                   pauliY, &target, 1, nullptr, nullptr, 0));
            CSV_CHECK(custatevecApplyPauliRotation(ctx.handle, ctx.d_state, CUDA_C_64F, kQ, tz,
                                                   pauliZ, &target, 1, nullptr, nullptr, 0));
        }
        for (int i = 0; i < kQ - 1; ++i) {
            const int32_t control = i;
            const int32_t target = i + 1;
            CSV_CHECK(custatevecApplyMatrix(ctx.handle, ctx.d_state, CUDA_C_64F, kQ, kMatrixX,
                                            CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, &target, 1,
                                            &control, nullptr, 1, CUSTATEVEC_COMPUTE_DEFAULT,
                                            ctx.extra, ctx.extra_size));
        }
    }

    static const custatevecPauli_t z_one[] = {CUSTATEVEC_PAULI_Z};
    static const custatevecPauli_t* pauli_exp[kQ] = {
        z_one, z_one, z_one, z_one, z_one, z_one, z_one, z_one,
        z_one, z_one, z_one, z_one, z_one, z_one, z_one, z_one};
    static const int32_t basis_bits[kQ] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    static const int32_t* basis_arr[kQ] = {
        basis_bits + 0,  basis_bits + 1,  basis_bits + 2,  basis_bits + 3,
        basis_bits + 4,  basis_bits + 5,  basis_bits + 6,  basis_bits + 7,
        basis_bits + 8,  basis_bits + 9,  basis_bits + 10, basis_bits + 11,
        basis_bits + 12, basis_bits + 13, basis_bits + 14, basis_bits + 15};
    static const uint32_t n_basis[kQ] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    CSV_CHECK(custatevecComputeExpectationsOnPauliBasis(
        ctx.handle, ctx.d_state, CUDA_C_64F, kQ, expectations,
        const_cast<const custatevecPauli_t**>(pauli_exp), kQ,
        const_cast<const int32_t**>(basis_arr), n_basis));
    return 0;
}

} // namespace

namespace {

/** Scoreboard for one of our simulators against cuStateVec. */
struct Score {
    const char* name;
    uint64_t fixed_mismatch = 0;
    uint64_t fixed_mismatch_real = 0; /* excluding exact boundary ties */
    uint64_t raw_mismatch = 0;
    double max_abs_diff = 0;
};

void score_case(Score& s, const double ours[kQ], const double ref[kQ], double tol,
                const uint8_t nibbles[QHASH_NIBBLE_COUNT], int offset)
{
    for (int q = 0; q < kQ; ++q) {
        const double d = std::fabs(ours[q] - ref[q]);
        if (d > s.max_abs_diff)
            s.max_abs_diff = d;
        if (d > tol)
            ++s.raw_mismatch;
        if (qhash::to_fixed_q15(ours[q]) == qhash::to_fixed_q15(ref[q]))
            continue;
        ++s.fixed_mismatch;
        if (!qhash_test::q15_is_tie(nibbles, offset, q, nullptr, nullptr))
            ++s.fixed_mismatch_real;
    }
}

} // namespace

int main(int argc, char** argv)
{
    uint32_t nonces = 1024;
    uint32_t nTime = QHASH_SF_ANGLE;
    uint32_t nonce_start = 0;
    int header_fill = 0;
    double tol = 1e-10;
    bool uniform = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--nonces" && i + 1 < argc)
            nonces = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--start" && i + 1 < argc)
            nonce_start = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--fill" && i + 1 < argc)
            header_fill = int(std::strtol(argv[++i], nullptr, 0));
        else if (a == "--ntime" && i + 1 < argc)
            nTime = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--tol" && i + 1 < argc)
            tol = std::strtod(argv[++i], nullptr);
        else if (a == "--uniform")
            uniform = true;
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--nonces N] [--start N] [--fill BYTE] [--ntime T] "
                        "[--tol EPS] [--uniform]\n"
                        "  --start/--fill reproduce a specific case from "
                        "qhash-closed-form-check (which fills the header with 0xA5)\n",
                        argv[0]);
            return 0;
        }
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::fprintf(stderr, "No CUDA device\n");
        return 1;
    }

    CsvCtx ctx;
    if (csv_init(ctx) != 0)
        return 1;

    uint8_t header[QHASH_INPUT_SIZE];
    std::memset(header, header_fill, sizeof(header));
    header[68] = uint8_t(nTime);
    header[69] = uint8_t(nTime >> 8);
    header[70] = uint8_t(nTime >> 16);
    header[71] = uint8_t(nTime >> 24);

    Score sv{"FP64 statevector"};
    Score cf{"closed form"};
    uint64_t qubit_checks = 0;
    uint64_t both_disagree = 0;

    const uint32_t cases = uniform ? (16u * 16u * 16u) : nonces;
    for (uint32_t n = 0; n < cases; ++n) {
        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        if (uniform) {
            /* Same (ny1, nz1, ny2) on every qubit — the degenerate family where
               ⟨Z⟩ can be exactly 2^-16, i.e. exactly on a Q1.15 boundary. */
            for (int q = 0; q < kQ; ++q) {
                nibbles[q] = uint8_t(n & 0xF);
                nibbles[kQ + q] = uint8_t((n >> 4) & 0xF);
                nibbles[2 * kQ + q] = uint8_t((n >> 8) & 0xF);
                nibbles[3 * kQ + q] = uint8_t((n + q) & 0xF);
            }
        } else {
            const uint32_t nonce = nonce_start + n;
            header[76] = uint8_t(nonce);
            header[77] = uint8_t(nonce >> 8);
            header[78] = uint8_t(nonce >> 16);
            header[79] = uint8_t(nonce >> 24);
            uint8_t hash[QHASH_SHA256_SIZE];
            qhash::sha256(header, QHASH_INPUT_SIZE, hash);
            qhash::split_nibbles(hash, nibbles);
        }

        double sv_exp[kQ], cf_exp[kQ], csv_exp[kQ];
        qhash_simulate_cpu_sim(nibbles, sv_exp, QHASH_PRECISION_FP64, nTime,
                               QHASH_SIM_STATEVECTOR);
        qhash_simulate_cpu_sim(nibbles, cf_exp, QHASH_PRECISION_FP64, nTime,
                               QHASH_SIM_CLOSED_FORM);
        if (csv_simulate(ctx, nibbles, nTime, csv_exp) != 0) {
            csv_destroy(ctx);
            return 1;
        }

        const int offset = qhash::angle_offset(nTime);
        const uint64_t sv_before = sv.fixed_mismatch;
        const uint64_t cf_before = cf.fixed_mismatch;
        score_case(sv, sv_exp, csv_exp, tol, nibbles, offset);
        score_case(cf, cf_exp, csv_exp, tol, nibbles, offset);
        qubit_checks += kQ;

        if (sv.fixed_mismatch != sv_before && cf.fixed_mismatch != cf_before)
            ++both_disagree;
        if (sv.fixed_mismatch != sv_before || cf.fixed_mismatch != cf_before) {
            for (int q = 0; q < kQ; ++q) {
                const int16_t a = qhash::to_fixed_q15(csv_exp[q]);
                const int16_t b = qhash::to_fixed_q15(sv_exp[q]);
                const int16_t c = qhash::to_fixed_q15(cf_exp[q]);
                if (a == b && a == c)
                    continue;
                long double exact = 0, margin = 0;
                const bool tie = qhash_test::q15_is_tie(nibbles, offset, q, &exact, &margin);
                std::printf("  case %u qubit %d: cuStateVec=%d statevector=%d closed=%d  "
                            "exact=%.21Lg margin=%.3Le %s\n",
                            n, q, int(a), int(b), int(c), exact, margin,
                            tie ? "-> EXACT TIE (no FP64 answer is defined)" : "-> REAL DIVERGENCE");
            }
        }

        if ((n + 1) % 256 == 0 || n + 1 == cases) {
            std::printf("progress %u / %u  statevector mm=%llu  closed-form mm=%llu\n", n + 1,
                        cases, (unsigned long long)sv.fixed_mismatch,
                        (unsigned long long)cf.fixed_mismatch);
            std::fflush(stdout);
        }
    }

    csv_destroy(ctx);

    std::printf("\nvs cuStateVec FP64 on %u %s (nTime=%u)\n", cases,
                uniform ? "uniform patterns" : "nonces", nTime);
    for (const Score* s : {&sv, &cf}) {
        std::printf("  %-17s Q1.15 mismatches %llu / %llu (%llu real, rest exact ties)   "
                    "raw |d|>%.1e: %llu   max|Δ⟨Z⟩| %.6e\n",
                    s->name, (unsigned long long)s->fixed_mismatch,
                    (unsigned long long)qubit_checks,
                    (unsigned long long)s->fixed_mismatch_real, tol,
                    (unsigned long long)s->raw_mismatch, s->max_abs_diff);
    }
    std::printf("  cases where both of ours disagree with cuStateVec: %llu\n",
                (unsigned long long)both_disagree);

    /* Only genuine divergence is a failure. On an exact tie cuStateVec is picking a
       side by rounding noise, so "matching the node" is not a defined property. */
    if (cf.fixed_mismatch_real != 0 || sv.fixed_mismatch_real != 0) {
        std::fprintf(stderr,
                     "FAIL: Q1.15 divergence vs cuStateVec away from a boundary tie "
                     "(statevector %llu, closed form %llu)\n",
                     (unsigned long long)sv.fixed_mismatch_real,
                     (unsigned long long)cf.fixed_mismatch_real);
        return 1;
    }
    if (cf.fixed_mismatch == 0 && sv.fixed_mismatch == 0) {
        std::printf("PASS: both simulators bit-exact in Q1.15 vs cuStateVec FP64\n");
        return 0;
    }
    std::printf("PASS: all %llu statevector / %llu closed-form disagreements are exact boundary "
                "ties; closed form is %.0fx closer to cuStateVec in raw ⟨Z⟩\n",
                (unsigned long long)sv.fixed_mismatch, (unsigned long long)cf.fixed_mismatch,
                sv.max_abs_diff / (cf.max_abs_diff > 0 ? cf.max_abs_diff : 1.0));
    return 0;
}
