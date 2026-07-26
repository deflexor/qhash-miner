/**
 * qhash-miner entry: benchmark / single-hash / smoke mine loop / Phase-4 compare.
 *
 * Correct algorithm (from official sources — differs from some public writeups):
 *   SHA256 → nibble angles → 16q/2-layer Ry/Rz/CNOT → ⟨Z⟩ Q1.15 → SHA256
 */
#include "qhash_cpu.h"
#include "qhash_kernel.cuh"
#include "qhash_params.h"
#include "circuit.cuh"
#include "sha256.cuh"
#include "target.cuh"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char* argv0)
{
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s --benchmark [--nonces N] [--threads T] [--blocks B] [--sim S]\n"
                 "                 [--chunk C] [--streams S] [--fp32] [--ntime T]\n"
                 "  %s --compare-fp [--nonces N] [--q15-nonces M] [--chunk C] [--threads T]\n"
                 "                 [--ntime T]  (Phase 4: digests×N, Q1.15×M default min(N,100k))\n"
                 "  %s --hash-hex <160 hex chars of 80-byte header>\n"
                 "  %s --self-test\n"
                 "\n"
                 "  --sim closed|statevector  simulator (default closed form)\n"
                 "  --threads T               32..1024 for the closed form; 128|256 statevector\n"
                 "  --blocks B                closed-form grid size; 0 = fill the device\n"
                 "  --chunk C                 nonces per launch; 0 = whole batch in one launch\n"
                 "                            (statevector default 128, it needs ~1 MiB/nonce)\n"
                 "  --streams S               statevector path only\n"
                 "\n"
                 "  Sustained closed-form rate needs a multi-second load, e.g.\n"
                 "    --benchmark --nonces 4000000000 --chunk 4194304\n"
                 "  A single cold launch reads low: the GPU has not boosted yet.\n",
                 argv0, argv0, argv0, argv0);
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

static bool parse_hex(const char* hex, uint8_t* out, size_t nbytes)
{
    if (std::strlen(hex) != nbytes * 2)
        return false;
    for (size_t i = 0; i < nbytes; ++i) {
        const int hi = hex_nibble(hex[2 * i]);
        const int lo = hex_nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0)
            return false;
        out[i] = uint8_t((hi << 4) | lo);
    }
    return true;
}

static void print_hex(const uint8_t* p, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        std::printf("%02x", p[i]);
}

static int self_test()
{
    uint8_t header[QHASH_INPUT_SIZE];
    std::memset(header, 0, sizeof(header));
    /* nTime after soft-fork angle activation */
    const uint32_t nTime = QHASH_SF_ANGLE;
    header[68] = uint8_t(nTime);
    header[69] = uint8_t(nTime >> 8);
    header[70] = uint8_t(nTime >> 16);
    header[71] = uint8_t(nTime >> 24);

    uint8_t h64[32], h32[32], hcf[32];
    qhash_hash_cpu_sim(header, h64, QHASH_PRECISION_FP64, nTime, QHASH_SIM_STATEVECTOR);
    qhash_hash_cpu(header, h32, QHASH_PRECISION_FP32, nTime);
    qhash_hash_cpu_sim(header, hcf, QHASH_PRECISION_FP64, nTime, QHASH_SIM_CLOSED_FORM);

    std::printf("self-test zero-header FP64: ");
    print_hex(h64, 32);
    std::printf("\nself-test zero-header FP32: ");
    print_hex(h32, 32);
    std::printf("\nself-test zero-header closed form: ");
    print_hex(hcf, 32);
    std::printf("\n");
    if (std::memcmp(h64, hcf, 32) != 0) {
        std::fprintf(stderr, "FAIL: CPU closed form != CPU statevector\n");
        return 1;
    }

    /* Determinism */
    uint8_t h64b[32];
    qhash_hash_cpu_sim(header, h64b, QHASH_PRECISION_FP64, nTime, QHASH_SIM_STATEVECTOR);
    if (std::memcmp(h64, h64b, 32) != 0) {
        std::fprintf(stderr, "FAIL: FP64 non-deterministic\n");
        return 1;
    }

    /* Nibble→expectation path: all-zero nibbles must be finite ∈ [-1,1] */
    uint8_t nibbles[QHASH_NIBBLE_COUNT];
    std::memset(nibbles, 0, sizeof(nibbles));
    double expv[QHASH_NUM_QUBITS];
    qhash_simulate_cpu(nibbles, expv, QHASH_PRECISION_FP64, nTime);
    for (int i = 0; i < QHASH_NUM_QUBITS; ++i) {
        if (!(expv[i] >= -1.0000001 && expv[i] <= 1.0000001)) {
            std::fprintf(stderr, "FAIL: expectation[%d]=%g out of range\n", i, expv[i]);
            return 1;
        }
    }
    std::printf("self-test expectations[0..3]: %g %g %g %g\n", expv[0], expv[1], expv[2],
                expv[3]);
    std::printf("CUDA available: %s\n", qhash_cuda_available() ? "yes" : "no");

    /* When CUDA is present, bit-compare GPU digests vs CPU reference (FP64). */
    if (qhash_cuda_available()) {
        uint8_t hgpu[32];
        if (qhash_hash_gpu(header, hgpu, QHASH_PRECISION_FP64, nTime) != 0) {
            std::fprintf(stderr, "FAIL: qhash_hash_gpu\n");
            return 1;
        }
        std::printf("self-test GPU FP64:      ");
        print_hex(hgpu, 32);
        std::printf("\n");
        if (std::memcmp(h64, hgpu, 32) != 0) {
            std::fprintf(stderr, "FAIL: GPU FP64 digest != CPU FP64\n");
            return 1;
        }

        /* Small multi-nonce golden: GPU batch digests must match CPU. */
        constexpr uint32_t kGold = 64;
        qhash_job_t job{};
        std::memset(job.header, 0, QHASH_INPUT_SIZE);
        job.header[68] = uint8_t(nTime);
        job.header[69] = uint8_t(nTime >> 8);
        job.header[70] = uint8_t(nTime >> 16);
        job.header[71] = uint8_t(nTime >> 24);
        job.nonce_start = 0;
        job.nonce_count = kGold;
        job.nTime = nTime;
        job.precision = QHASH_PRECISION_FP64;
        job.check_target = 0;
        std::memset(job.target, 0xFF, 32);

        /* Oracle digests for nonces 0..kGold-1, straight from the statevector. */
        std::vector<std::array<uint8_t, 32>> oracle(kGold);
        for (uint32_t i = 0; i < kGold; ++i) {
            uint8_t hdr[QHASH_INPUT_SIZE];
            std::memcpy(hdr, job.header, QHASH_INPUT_SIZE);
            hdr[76] = uint8_t(i);
            hdr[77] = uint8_t(i >> 8);
            hdr[78] = uint8_t(i >> 16);
            hdr[79] = uint8_t(i >> 24);
            qhash_hash_cpu_sim(hdr, oracle[i].data(), QHASH_PRECISION_FP64, nTime,
                               QHASH_SIM_STATEVECTOR);
        }

        struct GoldCfg {
            qhash_sim_t sim;
            int threads;
            int streams;
        };
        const GoldCfg cfgs[] = {
            {QHASH_SIM_CLOSED_FORM, 256, 1},
            {QHASH_SIM_CLOSED_FORM, 64, 1},
            {QHASH_SIM_STATEVECTOR, 256, 1},
            {QHASH_SIM_STATEVECTOR, 128, 2}, /* also exercise multi-stream once */
        };
        for (const GoldCfg& cfg : cfgs) {
            const int tc = cfg.threads;
            job.sim = cfg.sim;
            job.threads_per_block = tc;
            job.num_streams = cfg.streams;
            std::vector<qhash_share_t> shares(kGold);
            uint32_t share_count = 0;
            double secs = 0;
            if (qhash_mine_batch(&job, shares.data(), kGold, &share_count, &secs) != 0 ||
                share_count != kGold) {
                std::fprintf(stderr, "FAIL: GPU gold batch threads=%d (got %u shares)\n", tc,
                             share_count);
                return 1;
            }
            for (uint32_t i = 0; i < kGold; ++i) {
                const qhash_share_t* s = nullptr;
                for (uint32_t j = 0; j < share_count; ++j) {
                    if (shares[j].nonce == i) {
                        s = &shares[j];
                        break;
                    }
                }
                if (!s || std::memcmp(s->hash, oracle[i].data(), 32) != 0) {
                    std::fprintf(stderr, "FAIL: GPU!=CPU statevector at nonce %u threads=%d\n", i,
                                 tc);
                    return 1;
                }
            }
            std::printf("self-test GPU %-11s vs CPU statevector: %u nonces bit-exact OK "
                        "(threads=%d streams=%d, %.3f s)\n",
                        cfg.sim == QHASH_SIM_CLOSED_FORM ? "closed-form" : "statevector", kGold, tc,
                        job.num_streams, secs);
        }

        /* Phase 6.12: mine with a target loose enough to produce candidates, and
           confirm each returned share carries the oracle's digest and clears the
           target. A leading target byte of 0x04 hits roughly 1 nonce in 64. */
        {
            constexpr uint32_t kMine = 4096;
            job.sim = QHASH_SIM_CLOSED_FORM;
            job.threads_per_block = 256;
            job.num_streams = 1;
            job.blocks = 0;
            job.nonce_start = 0;
            job.nonce_count = kMine;
            job.check_target = 1;
            /* Little-endian target (Bitcoin/cpuminer layout): byte 31 is the most
               significant, so this asks for a digest whose top byte is below 0x04. */
            std::memset(job.target, 0xFF, 32);
            job.target[31] = 0x04;

            std::vector<qhash_share_t> shares(kMine);
            uint32_t found = 0;
            double secs = 0;
            if (qhash_mine_batch(&job, shares.data(), kMine, &found, &secs) != 0) {
                std::fprintf(stderr, "FAIL: candidate mining batch\n");
                return 1;
            }
            for (uint32_t i = 0; i < found; ++i) {
                uint8_t hdr[QHASH_INPUT_SIZE];
                std::memcpy(hdr, job.header, QHASH_INPUT_SIZE);
                const uint32_t n = shares[i].nonce;
                hdr[76] = uint8_t(n);
                hdr[77] = uint8_t(n >> 8);
                hdr[78] = uint8_t(n >> 16);
                hdr[79] = uint8_t(n >> 24);
                uint8_t sv[32];
                qhash_hash_cpu_sim(hdr, sv, QHASH_PRECISION_FP64, nTime, QHASH_SIM_STATEVECTOR);
                if (std::memcmp(shares[i].hash, sv, 32) != 0) {
                    std::fprintf(stderr, "FAIL: share %u digest is not the oracle's\n", n);
                    return 1;
                }
                if (!qhash_hash_le_target(shares[i].hash, job.target)) {
                    std::fprintf(stderr, "FAIL: share %u does not meet the target\n", n);
                    return 1;
                }
            }
            uint64_t checked = 0, rejected = 0;
            qhash_cuda_reverify_stats(&checked, &rejected);
            if (found == 0 || checked < found) {
                std::fprintf(stderr, "FAIL: re-verification did not run (%u shares, %llu checked)\n",
                             found, (unsigned long long)checked);
                return 1;
            }
            std::printf("self-test candidate re-verification: %u shares, %llu oracle checks, "
                        "%llu rejected\n",
                        found, (unsigned long long)checked, (unsigned long long)rejected);
        }
    }

    std::printf("self-test OK\n");
    return 0;
}

static int run_benchmark(uint32_t nonces, qhash_precision_t prec, uint32_t nTime, uint32_t chunk,
                         int threads, int streams, qhash_sim_t sim, int blocks)
{
    qhash_job_t job{};
    std::memset(job.header, 0xA5, QHASH_INPUT_SIZE);
    job.header[68] = uint8_t(nTime);
    job.header[69] = uint8_t(nTime >> 8);
    job.header[70] = uint8_t(nTime >> 16);
    job.header[71] = uint8_t(nTime >> 24);
    job.nTime = nTime;
    job.precision = prec;
    job.threads_per_block = threads;
    job.num_streams = streams;
    job.sim = sim;
    job.blocks = blocks;

    /* An unreachable target keeps the share queue empty, so the measurement is not
       one atomicAdd per nonce on a single counter. */
    job.check_target = 1;
    std::memset(job.target, 0x00, 32);

    qhash_share_t share{};
    uint32_t share_count = 0;
    double seconds = 0;

    /* The closed-form kernel holds no per-nonce state, so a batch is one launch.
       The statevector kernel needs ~1 MiB per nonce and must stay chunked. */
    const bool statevector = (sim == QHASH_SIM_STATEVECTOR) || (prec == QHASH_PRECISION_FP32);
    if (chunk == 0)
        chunk = statevector ? 128 : nonces;

    uint32_t done = 0;
    double total_s = 0;
    while (done < nonces) {
        job.nonce_start = done;
        job.nonce_count = (nonces - done > chunk) ? chunk : (nonces - done);
        share_count = 0;
        if (qhash_mine_batch(&job, &share, 1, &share_count, &seconds) != 0) {
            std::fprintf(stderr, "mine_batch failed\n");
            return 1;
        }
        total_s += seconds;
        done += job.nonce_count;
    }

    const double hs = (total_s > 0) ? (double(nonces) / total_s) : 0;
    std::printf("Nonces tested: %u\n", nonces);
    std::printf("Time: %.4f seconds\n", total_s);
    std::printf("Hashrate: %.2f H/s (%.3f kh/s, %.3f Mh/s)\n", hs, hs / 1e3, hs / 1e6);
    std::printf("Backend: %s (%d SMs)  sim: %s  precision: %s  threads: %d  blocks: %d",
                qhash_cuda_available() ? "CUDA" : "CPU", qhash_cuda_sm_count(),
                statevector ? "statevector" : "closed-form",
                prec == QHASH_PRECISION_FP32 ? "FP32" : "FP64", threads, blocks);
    if (statevector)
        std::printf("  chunk: %u  streams: %d", chunk, streams);
    std::printf("\n");
    return 0;
}

/**
 * Phase 4: compare FP32 vs FP64.
 * - Digests over all `nonces` (GPU batches when available; consensus gate).
 * - Q1.15 fixed-point over `q15_nonces` on CPU (diagnostic; default min(N,100k)).
 * Gate FP32 mining only if digest match ≥ 99.99%.
 */
static int run_compare_fp(uint32_t nonces, uint32_t nTime, uint32_t chunk, int threads,
                          uint32_t q15_nonces)
{
    if (chunk == 0)
        chunk = 128;
    if (q15_nonces == 0 || q15_nonces > nonces)
        q15_nonces = nonces > 100000u ? 100000u : nonces;

    uint8_t header[QHASH_INPUT_SIZE];
    std::memset(header, 0xA5, sizeof(header));
    header[68] = uint8_t(nTime);
    header[69] = uint8_t(nTime >> 8);
    header[70] = uint8_t(nTime >> 16);
    header[71] = uint8_t(nTime >> 24);

    uint64_t qubit_checks = 0;
    uint64_t fixed_mismatch = 0;
    uint64_t digest_mismatch = 0;
    uint64_t nonce_q15_any = 0;
    int mismatch_hist[QHASH_NUM_QUBITS + 1];
    std::memset(mismatch_hist, 0, sizeof(mismatch_hist));

    const bool use_gpu = qhash_cuda_available() != 0;
    std::printf("Phase 4 FP32 vs FP64: digests=%u, Q1.15 sample=%u, nTime=%u, backend=%s\n",
                nonces, q15_nonces, nTime, use_gpu ? "GPU digests" : "CPU digests");

    /* ---- Q1.15 sample (CPU) ---- */
    for (uint32_t nonce = 0; nonce < q15_nonces; ++nonce) {
        header[76] = uint8_t(nonce);
        header[77] = uint8_t(nonce >> 8);
        header[78] = uint8_t(nonce >> 16);
        header[79] = uint8_t(nonce >> 24);

        uint8_t hash[32];
        qhash::sha256(header, QHASH_INPUT_SIZE, hash);
        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        qhash::split_nibbles(hash, nibbles);

        double e64[QHASH_NUM_QUBITS], e32[QHASH_NUM_QUBITS];
        qhash_simulate_cpu_sim(nibbles, e64, QHASH_PRECISION_FP64, nTime, QHASH_SIM_STATEVECTOR);
        qhash_simulate_cpu_sim(nibbles, e32, QHASH_PRECISION_FP32, nTime, QHASH_SIM_STATEVECTOR);

        int mm = 0;
        for (int q = 0; q < QHASH_NUM_QUBITS; ++q) {
            ++qubit_checks;
            if (qhash::to_fixed_q15(e64[q]) != qhash::to_fixed_q15(e32[q])) {
                ++fixed_mismatch;
                ++mm;
            }
        }
        ++mismatch_hist[mm];
        if (mm > 0)
            ++nonce_q15_any;

        if ((nonce + 1) % 1024 == 0 || nonce + 1 == q15_nonces) {
            std::printf("Q1.15 progress %u / %u  qubit mm=%llu/%llu  nonce-any mm=%llu\n",
                        nonce + 1, q15_nonces, (unsigned long long)fixed_mismatch,
                        (unsigned long long)qubit_checks, (unsigned long long)nonce_q15_any);
            std::fflush(stdout);
        }
    }

    /* ---- Full digest compare ---- */
    uint32_t done = 0;
    while (done < nonces) {
        const uint32_t n = (nonces - done > chunk) ? chunk : (nonces - done);

        if (use_gpu) {
            qhash_job_t job{};
            std::memcpy(job.header, header, QHASH_INPUT_SIZE);
            job.nonce_start = done;
            job.nonce_count = n;
            job.nTime = nTime;
            job.check_target = 0;
            job.threads_per_block = threads;
            job.num_streams = 1;
            /* Phase 4 asks how FP32 statevector rounding compares to FP64 statevector
               rounding, so both legs must stay on the statevector kernel. */
            job.sim = QHASH_SIM_STATEVECTOR;
            std::memset(job.target, 0xFF, 32);

            std::vector<qhash_share_t> shares64(n), shares32(n);
            uint32_t sc64 = 0, sc32 = 0;
            double secs = 0;

            job.precision = QHASH_PRECISION_FP64;
            if (qhash_mine_batch(&job, shares64.data(), n, &sc64, &secs) != 0 || sc64 != n) {
                std::fprintf(stderr, "FAIL: FP64 batch at nonce %u\n", done);
                return 1;
            }
            job.precision = QHASH_PRECISION_FP32;
            if (qhash_mine_batch(&job, shares32.data(), n, &sc32, &secs) != 0 || sc32 != n) {
                std::fprintf(stderr, "FAIL: FP32 batch at nonce %u\n", done);
                return 1;
            }

            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t nonce = done + i;
                const uint8_t* h64 = nullptr;
                const uint8_t* h32 = nullptr;
                for (uint32_t j = 0; j < sc64; ++j)
                    if (shares64[j].nonce == nonce) {
                        h64 = shares64[j].hash;
                        break;
                    }
                for (uint32_t j = 0; j < sc32; ++j)
                    if (shares32[j].nonce == nonce) {
                        h32 = shares32[j].hash;
                        break;
                    }
                if (!h64 || !h32 || std::memcmp(h64, h32, 32) != 0)
                    ++digest_mismatch;
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t nonce = done + i;
                header[76] = uint8_t(nonce);
                header[77] = uint8_t(nonce >> 8);
                header[78] = uint8_t(nonce >> 16);
                header[79] = uint8_t(nonce >> 24);
                uint8_t h64[32], h32[32];
                qhash_hash_cpu(header, h64, QHASH_PRECISION_FP64, nTime);
                qhash_hash_cpu(header, h32, QHASH_PRECISION_FP32, nTime);
                if (std::memcmp(h64, h32, 32) != 0)
                    ++digest_mismatch;
            }
        }

        done += n;
        if (done % 4096 == 0 || done == nonces) {
            const double dig_match =
                (done > 0) ? (100.0 * double(done - digest_mismatch) / double(done)) : 0;
            std::printf("digest progress %u / %u  mm=%llu  match=%.4f%%\n", done, nonces,
                        (unsigned long long)digest_mismatch, dig_match);
            std::fflush(stdout);
        }
    }

    const double dig_match_pct =
        (nonces > 0) ? (100.0 * double(nonces - digest_mismatch) / double(nonces)) : 0;
    const double q15_match_pct =
        (qubit_checks > 0)
            ? (100.0 * double(qubit_checks - fixed_mismatch) / double(qubit_checks))
            : 0;
    const double nonce_q15_match =
        (q15_nonces > 0) ? (100.0 * double(q15_nonces - nonce_q15_any) / double(q15_nonces)) : 0;

    std::printf("\n=== Phase 4 summary ===\n");
    std::printf("Q1.15 sample:       %u nonces\n", q15_nonces);
    std::printf("Q1.15 qubit match:  %.6f%%  (%llu mismatches / %llu qubits)\n", q15_match_pct,
                (unsigned long long)fixed_mismatch, (unsigned long long)qubit_checks);
    std::printf("Q1.15 nonce match:  %.6f%%  (%llu nonces with ≥1 qubit mismatch)\n",
                nonce_q15_match, (unsigned long long)nonce_q15_any);
    std::printf("Digest match:       %.6f%%  (%llu mismatches / %u)\n", dig_match_pct,
                (unsigned long long)digest_mismatch, nonces);
    std::printf("Mismatch-count hist (qubits wrong per nonce):");
    for (int i = 0; i <= QHASH_NUM_QUBITS; ++i) {
        if (mismatch_hist[i])
            std::printf(" [%d]=%d", i, mismatch_hist[i]);
    }
    std::printf("\n");

    if (dig_match_pct >= 99.99)
        std::printf("VERDICT: FP32 digest match ≥99.99%% — FP32 mining *candidate* (still review)\n");
    else
        std::printf("VERDICT: FP32 UNSAFE for consensus mining (need ≥99.99%% digest match)\n");
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    uint32_t nonces = 0; /* 0 → per-simulator default below */
    uint32_t chunk = 0; /* 0 → per-simulator default in run_benchmark */
    uint32_t q15_nonces = 0; /* 0 → auto min(N,100k) in --compare-fp */
    int threads = 0; /* 0 → per-simulator default below */
    int blocks = 0; /* 0 → fill the device */
    int streams = QHASH_DEFAULT_STREAMS;
    qhash_precision_t prec = QHASH_PRECISION_FP64;
    qhash_sim_t sim = QHASH_DEFAULT_SIM;
    uint32_t nTime = QHASH_SF_ANGLE; /* post-angle softfork default */

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--fp32")
            prec = QHASH_PRECISION_FP32;
        else if (a == "--sim" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "closed" || v == "closed-form" || v == "cf")
                sim = QHASH_SIM_CLOSED_FORM;
            else if (v == "statevector" || v == "sv")
                sim = QHASH_SIM_STATEVECTOR;
            else {
                std::fprintf(stderr, "--sim must be closed or statevector\n");
                return 1;
            }
        } else if (a == "--blocks" && i + 1 < argc)
            blocks = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--nonces" && i + 1 < argc)
            nonces = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--q15-nonces" && i + 1 < argc)
            q15_nonces = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--chunk" && i + 1 < argc)
            chunk = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--threads" && i + 1 < argc)
            threads = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--streams" && i + 1 < argc)
            streams = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--ntime" && i + 1 < argc)
            nTime = uint32_t(std::strtoul(argv[++i], nullptr, 10));
    }

    const bool statevector_path = (sim == QHASH_SIM_STATEVECTOR) || prec == QHASH_PRECISION_FP32;
    if (threads == 0)
        threads = statevector_path ? QHASH_DEFAULT_THREADS : QHASH_CF_DEFAULT_THREADS;
    if (nonces == 0)
        nonces = statevector_path ? 256 : (1u << 22);

    if (statevector_path) {
        if (threads != 128 && threads != 256) {
            std::fprintf(stderr, "--threads must be 128 or 256 for the statevector kernel\n");
            return 1;
        }
    } else if (threads < 32 || threads > 1024 || (threads % 32) != 0) {
        std::fprintf(stderr, "--threads must be a multiple of 32 in 32..1024\n");
        return 1;
    }
    if (blocks < 0) {
        std::fprintf(stderr, "--blocks must be >= 0\n");
        return 1;
    }
    if (streams < 1 || streams > QHASH_MAX_STREAMS) {
        std::fprintf(stderr, "--streams must be 1..%d\n", QHASH_MAX_STREAMS);
        return 1;
    }

    if (std::strcmp(argv[1], "--self-test") == 0)
        return self_test();

    if (std::strcmp(argv[1], "--benchmark") == 0)
        return run_benchmark(nonces, prec, nTime, chunk, threads, streams, sim, blocks);

    if (std::strcmp(argv[1], "--compare-fp") == 0)
        return run_compare_fp(nonces, nTime, chunk == 0 ? 128 : chunk,
                              (threads == 128 || threads == 256) ? threads : 256, q15_nonces);

    if (std::strcmp(argv[1], "--hash-hex") == 0) {
        if (argc < 3) {
            usage(argv[0]);
            return 1;
        }
        uint8_t header[QHASH_INPUT_SIZE];
        if (!parse_hex(argv[2], header, QHASH_INPUT_SIZE)) {
            std::fprintf(stderr, "expected %d hex bytes\n", QHASH_INPUT_SIZE);
            return 1;
        }
        uint8_t out[32];
        qhash_hash_cpu(header, out, prec, nTime);
        print_hex(out, 32);
        std::printf("\n");
        return 0;
    }

    usage(argv[0]);
    return 1;
}
