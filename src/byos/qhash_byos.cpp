/**
 * BYOS drop-in for official qubitcoin-miner (algo/qhash/qhash-gate.h).
 *
 * Replace algo/qhash/qhash-custatevec.c with this translation unit and link
 * against libqhash_core — see scripts/build-official-byos.sh / PLAN.md Phase 5.
 *
 * Defaults match the stock miner gate: FP32 + legacy angles (offset=0).
 * For node/consensus digests build with:
 *   -DQHASH_BYOS_FP64=1
 * and call qhash_byos_set_ntime() from qhash_hash (nTime from header).
 */
#include "qhash_cpu.h"
#include "qhash_params.h"

#include <cstring>

/* Mirror official miner gate header constants. */
#ifndef NUM_QUBITS
#define NUM_QUBITS QHASH_NUM_QUBITS
#endif
#ifndef SHA256_BLOCK_SIZE
#define SHA256_BLOCK_SIZE QHASH_SHA256_SIZE
#endif

#ifndef QHASH_BYOS_FP64
#define QHASH_BYOS_FP64 0
#endif
#ifndef QHASH_BYOS_NTIME
#define QHASH_BYOS_NTIME 0u
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if QHASH_BYOS_FP64
static __thread uint32_t g_byos_ntime = QHASH_BYOS_NTIME;

void qhash_byos_set_ntime(uint32_t nTime)
{
    g_byos_ntime = nTime;
}
#endif

bool qhash_thread_init(int thr_id)
{
    (void)thr_id;
    return true;
}

void run_simulation(const unsigned char data[2 * SHA256_BLOCK_SIZE],
                    double expectations[NUM_QUBITS])
{
#if QHASH_BYOS_FP64
    qhash_simulate_cpu(data, expectations, QHASH_PRECISION_FP64, g_byos_ntime);
#else
    /* Stock miner BYOS sample: FP32 + pre-softfork angles. */
    qhash_simulate_cpu(data, expectations, QHASH_PRECISION_FP32, 0 /* legacy angles */);
#endif
}

#ifdef __cplusplus
}
#endif
