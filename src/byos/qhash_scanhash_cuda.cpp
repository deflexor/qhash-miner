/**
 * Batched CUDA scanhash for the qhash BYOS drop-in (Phase 5, retuned in 6.11).
 *
 * Lives here so it is version-controlled with the kernel it drives;
 * scripts/build-official-byos.sh copies it into official-miner/algo/qhash/ as
 * qhash-scanhash-cuda.cpp. Nothing in official-miner's own sources is touched,
 * and the qhash_job_t / qhash_mine_batch ABI is used exactly as published: the
 * job is zero-initialised, so the fields Phase 6 appended keep their defaults
 * (closed-form simulator, occupancy-derived grid).
 *
 * Prefer -t 1: one GPU, and the mutex below serialises miner threads anyway.
 */
#include "qhash_kernel.cuh"
#include "qhash_params.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

extern "C" {
#include "qhash-gate.h"
#include "miner.h"
}

extern bool opt_benchmark;

namespace {

std::mutex g_gpu_mu;

/**
 * Nonces per kernel launch.
 *
 * The closed-form kernel retires a nonce in ~3.7 ns, so the fixed per-launch cost
 * (constant-memory upload, counter reset, event sync) is what decides throughput.
 * Measured on an RTX 5060 Laptop at 268 Mh/s: 64k nonces per launch loses ~45%,
 * 1M loses ~1.5%, and 4M is indistinguishable from an arbitrarily long launch
 * while still returning every ~15 ms, which keeps work_restart responsive.
 */
constexpr uint32_t kBatch = 1u << 22;

/** Candidates are rare at any real difficulty; this is only an overflow guard. */
constexpr uint32_t kMaxShares = 1024;

uint32_t header_ntime_le(const uint8_t header[QHASH_INPUT_SIZE])
{
    return uint32_t(header[68]) | (uint32_t(header[69]) << 8) | (uint32_t(header[70]) << 16) |
           (uint32_t(header[71]) << 24);
}

} // namespace

extern "C" int scanhash_qhash_cuda(struct work *work, uint32_t max_nonce, uint64_t *hashes_done,
                                   struct thr_info *mythr)
{
    uint32_t edata[20] __attribute__((aligned(64)));
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    const uint32_t last_nonce = max_nonce - 1;
    uint32_t n = first_nonce;
    const int thr_id = mythr->id;
    const bool bench = opt_benchmark;

    if (!qhash_cuda_available()) {
        /* Avoid a tight error loop when CUDA is missing / stubbed (e.g. WSL
         * without GPU passthrough). Fall back to CPU BYOS via scanhash_generic
         * → qhash_hash → run_simulation (FP64 when built with BYOS FP64). */
        static bool warned;
        if (!warned) {
            applog(LOG_WARNING, "qhash CUDA BYOS: no CUDA device; falling back to CPU BYOS");
            warned = true;
        }
        return scanhash_generic(work, max_nonce, hashes_done, mythr);
    }

    v128_bswap32_80(edata, pdata);

    std::vector<qhash_share_t> shares(kMaxShares);

    while (n <= last_nonce && !work_restart[thr_id].restart) {
        uint32_t count = last_nonce - n + 1;
        if (count > kBatch)
            count = kBatch;

        qhash_job_t job{};
        std::memcpy(job.header, edata, QHASH_INPUT_SIZE);
        job.nonce_start = n;
        job.nonce_count = count;
        job.nTime = header_ntime_le(reinterpret_cast<const uint8_t *>(edata));
        /* FP64 consensus path — Phase 4 rejected FP32 mining digests. */
        job.precision = QHASH_PRECISION_FP64;
        job.threads_per_block = QHASH_CF_DEFAULT_THREADS;
        job.num_streams = 1;

        /* Always target-filter. Benchmarking with "record everything" would make
           the measurement one atomic increment per nonce, so it uses an
           unreachable target instead and simply keeps the queue empty. */
        job.check_target = 1;
        if (bench)
            std::memset(job.target, 0x00, QHASH_SHA256_SIZE);
        else
            std::memcpy(job.target, ptarget, QHASH_SHA256_SIZE);

        uint32_t share_count = 0;
        double secs = 0;
        int rc;
        {
            std::lock_guard<std::mutex> lock(g_gpu_mu);
            rc = qhash_mine_batch(&job, shares.data(), uint32_t(shares.size()), &share_count,
                                  &secs);
        }
        if (rc != 0) {
            applog(LOG_ERR, "qhash CUDA BYOS: qhash_mine_batch failed (%d)", rc);
            break;
        }

        if (!bench && share_count > 0) {
            /* Digests here have already been re-derived by the statevector oracle
               inside qhash_mine_batch (Phase 6.12), so this is the ordinary
               belt-and-braces check the miner does for every algorithm. */
            for (uint32_t i = 0; i < share_count; ++i) {
                uint32_t hash_words[8];
                std::memcpy(hash_words, shares[i].hash, 32);
                if (valid_hash(hash_words, ptarget)) {
                    pdata[19] = bswap_32(shares[i].nonce);
                    submit_solution(work, hash_words, mythr);
                }
            }
        }

        n += count;
    }

    *hashes_done = n - first_nonce;
    pdata[19] = n;
    return 0;
}
