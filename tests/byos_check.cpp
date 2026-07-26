/**
 * BYOS ABI smoke: run_simulation matches qhash_simulate_cpu for stock defaults
 * (FP32, legacy angles) and optional FP64 build.
 */
#include "qhash_cpu.h"
#include "qhash_params.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef NUM_QUBITS
#define NUM_QUBITS QHASH_NUM_QUBITS
#endif
#ifndef SHA256_BLOCK_SIZE
#define SHA256_BLOCK_SIZE QHASH_SHA256_SIZE
#endif

extern "C" bool qhash_thread_init(int thr_id);
extern "C" void run_simulation(const unsigned char data[2 * SHA256_BLOCK_SIZE],
                               double expectations[NUM_QUBITS]);

static int fail = 0;

static void check_vec(const char* tag, const double* a, const double* b)
{
    for (int i = 0; i < QHASH_NUM_QUBITS; ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > 1e-12) {
            std::printf("FAIL %s qubit %d |Δ|=%.3e  byos=%.17g cpu=%.17g\n", tag, i, d, a[i],
                        b[i]);
            fail = 1;
        }
    }
}

int main()
{
    if (!qhash_thread_init(0)) {
        std::printf("qhash_thread_init failed\n");
        return 1;
    }

    uint8_t nibbles[QHASH_NIBBLE_COUNT];
    std::memset(nibbles, 0, sizeof(nibbles));
    for (int i = 0; i < QHASH_NIBBLE_COUNT; ++i)
        nibbles[i] = uint8_t(i & 0xF);

    double byos[QHASH_NUM_QUBITS];
    double cpu[QHASH_NUM_QUBITS];
    run_simulation(nibbles, byos);

#if defined(QHASH_BYOS_FP64) && QHASH_BYOS_FP64
    qhash_simulate_cpu(nibbles, cpu, QHASH_PRECISION_FP64, 0);
    check_vec("FP64", byos, cpu);
#else
    qhash_simulate_cpu(nibbles, cpu, QHASH_PRECISION_FP32, 0);
    check_vec("FP32-legacy", byos, cpu);
#endif

    /* Second vector: all 0xA nibbles. */
    std::memset(nibbles, 0xA, sizeof(nibbles));
    run_simulation(nibbles, byos);
#if defined(QHASH_BYOS_FP64) && QHASH_BYOS_FP64
    qhash_simulate_cpu(nibbles, cpu, QHASH_PRECISION_FP64, 0);
    check_vec("FP64-A", byos, cpu);
#else
    qhash_simulate_cpu(nibbles, cpu, QHASH_PRECISION_FP32, 0);
    check_vec("FP32-A", byos, cpu);
#endif

    if (fail) {
        std::printf("byos_check FAILED\n");
        return 1;
    }
    std::printf("byos_check OK (ABI + expectations match CPU ref)\n");
    return 0;
}
