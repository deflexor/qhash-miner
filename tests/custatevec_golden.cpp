/**
 * Golden harness: cuStateVec FP64 (node consensus angles) vs CPU reference.
 *
 * Builds only when QHASH_ENABLE_CUSTATEVEC=ON and cuQuantum + cuBLAS 13 are found.
 * Compares Q1.15 fixed-points (and raw ⟨Z⟩ within tol) on ≥1000 nonces by default.
 *
 * Usage:
 *   qhash-custatevec-golden [--nonces N] [--ntime T] [--tol EPS]
 */
#include "circuit.cuh"
#include "qhash_cpu.h"
#include "qhash_params.h"
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

int main(int argc, char** argv)
{
    uint32_t nonces = 1024;
    uint32_t nTime = QHASH_SF_ANGLE;
    double tol = 1e-10;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--nonces" && i + 1 < argc)
            nonces = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--ntime" && i + 1 < argc)
            nTime = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--tol" && i + 1 < argc)
            tol = std::strtod(argv[++i], nullptr);
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--nonces N] [--ntime T] [--tol EPS]\n", argv[0]);
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
    std::memset(header, 0, sizeof(header));
    header[68] = uint8_t(nTime);
    header[69] = uint8_t(nTime >> 8);
    header[70] = uint8_t(nTime >> 16);
    header[71] = uint8_t(nTime >> 24);

    uint64_t fixed_mismatch = 0;
    uint64_t raw_mismatch = 0;
    uint64_t qubit_checks = 0;
    double max_abs_diff = 0;

    for (uint32_t n = 0; n < nonces; ++n) {
        header[76] = uint8_t(n);
        header[77] = uint8_t(n >> 8);
        header[78] = uint8_t(n >> 16);
        header[79] = uint8_t(n >> 24);

        uint8_t hash[QHASH_SHA256_SIZE];
        qhash::sha256(header, QHASH_INPUT_SIZE, hash);
        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        qhash::split_nibbles(hash, nibbles);

        double cpu_exp[kQ], csv_exp[kQ];
        qhash_simulate_cpu(nibbles, cpu_exp, QHASH_PRECISION_FP64, nTime);
        if (csv_simulate(ctx, nibbles, nTime, csv_exp) != 0) {
            csv_destroy(ctx);
            return 1;
        }

        for (int q = 0; q < kQ; ++q) {
            ++qubit_checks;
            const double d = std::fabs(cpu_exp[q] - csv_exp[q]);
            if (d > max_abs_diff)
                max_abs_diff = d;
            if (d > tol)
                ++raw_mismatch;
            if (qhash::to_fixed_q15(cpu_exp[q]) != qhash::to_fixed_q15(csv_exp[q]))
                ++fixed_mismatch;
        }

        if ((n + 1) % 256 == 0 || n + 1 == nonces)
            std::printf("progress %u / %u  fixed_mismatch=%llu  raw_mismatch=%llu  max|d|=%.3e\n",
                        n + 1, nonces, (unsigned long long)fixed_mismatch,
                        (unsigned long long)raw_mismatch, max_abs_diff);
    }

    csv_destroy(ctx);

    std::printf("cuStateVec FP64 vs CPU FP64 on %u nonces (nTime=%u)\n", nonces, nTime);
    std::printf("  Q1.15 mismatches: %llu / %llu\n", (unsigned long long)fixed_mismatch,
                (unsigned long long)qubit_checks);
    std::printf("  raw |d|>%.1e:     %llu / %llu\n", tol, (unsigned long long)raw_mismatch,
                (unsigned long long)qubit_checks);
    std::printf("  max |Δ⟨Z⟩|:       %.6e\n", max_abs_diff);

    if (fixed_mismatch != 0) {
        std::fprintf(stderr, "FAIL: Q1.15 mismatch vs cuStateVec\n");
        return 1;
    }
    std::printf("PASS: bit-exact Q1.15 vs cuStateVec FP64\n");
    return 0;
}
