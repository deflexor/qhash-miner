/**
 * Phase 6.11/6.14 profile: what actually limits the closed-form kernel.
 *
 * Nsight Compute needs GPU performance counters, which are not available on this
 * host (ERR_NVGPUCTRPERM under WSL). Ablation gives the same answer for the one
 * question that matters — is the kernel FP64-bound or SHA-bound — by timing the
 * production inner loop with one half removed:
 *
 *   full   both halves, identical arithmetic to the shipped kernel
 *   sha    the three SHA-256 compressions, ⟨Z⟩ replaced by a constant
 *   sweep  the 16-step ⟨Z⟩ sweep, nibbles from a cheap integer mix
 *   empty  loop and share-queue overhead only
 *
 * 1/t_full ≈ 1/t_sha + 1/t_sweep would mean the two halves do not overlap; a
 * t_full close to max(t_sha, t_sweep) means they do.
 */
#include "closed_form.cuh"
#include "qhash_kernel.cuh"
#include "sha256_mine.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace qhash;

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));        \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

namespace {

struct Params {
    uint32_t midstate[8];
    uint32_t hdr_w[3];
    uint32_t pad_w[64];
    AngleLut lut;
    uint32_t nTime;
};

__constant__ Params c_p;

__device__ __forceinline__ uint32_t bswap32(uint32_t x)
{
    return ((x & 0x000000FFu) << 24) | ((x & 0x0000FF00u) << 8) | ((x & 0x00FF0000u) >> 8) |
           ((x & 0xFF000000u) >> 24);
}

/** Header hash for one nonce: job midstate plus the nonce-dependent block. */
__device__ __forceinline__ void header_hash(uint32_t nonce, uint32_t hs[8])
{
#pragma unroll
    for (int i = 0; i < 8; ++i)
        hs[i] = c_p.midstate[i];
    uint32_t w[16];
    w[0] = c_p.hdr_w[0];
    w[1] = c_p.hdr_w[1];
    w[2] = c_p.hdr_w[2];
    w[3] = bswap32(nonce);
    w[4] = 0x80000000u;
#pragma unroll
    for (int i = 5; i < 15; ++i)
        w[i] = 0;
    w[15] = QHASH_INPUT_SIZE * 8;
    sha256_compress(hs, w);
}

/** Fold a digest into the sink so nothing can be dead-code eliminated. */
__device__ __forceinline__ void sink_digest(const uint32_t d[8], uint32_t* sink)
{
    uint32_t acc = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        acc ^= d[i];
    if (acc == 0xDEADBEEFu) /* essentially never true */
        atomicAdd(sink, acc);
}

enum Variant { kFull = 0, kSha = 1, kSweep = 2, kEmpty = 3, kProd = 4, kVariants = 5 };

template <int V>
__global__ void k_ablate(uint32_t nonce_start, uint32_t nonce_count, uint32_t* sink)
{
    __shared__ AngleLut s_lut;
    {
        double* dst = reinterpret_cast<double*>(&s_lut);
        const double* src = reinterpret_cast<const double*>(&c_p.lut);
        for (int i = int(threadIdx.x); i < QHASH_ANGLE_LUT_DOUBLES; i += int(blockDim.x))
            dst[i] = src[i];
    }
    __syncthreads();

    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nonce_count; idx += stride) {
        const uint32_t nonce = nonce_start + idx;

        if (V == kEmpty) {
            uint32_t d[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                d[i] = nonce + uint32_t(i);
            sink_digest(d, sink);
            continue;
        }

        uint32_t hs[8];
        double e[QHASH_NUM_QUBITS];

        if (V == kSweep) {
            /* No SHA at all: mix the nonce into six words with cheap integer ops,
               so the sweep still sees data-dependent nibbles. */
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t x = nonce * (0x9E3779B9u + 0x85EBCA6Bu * uint32_t(i)) + 0xC2B2AE35u;
                x ^= x >> 15;
                x *= 0x2545F491u;
                hs[i] = x ^ (x >> 13);
            }
            closed_form_sweep(s_lut, NibbleWords{hs}, e);
            uint32_t d[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                d[i] = uint32_t(int32_t(to_fixed_q15(e[2 * i]))) ^
                       (uint32_t(int32_t(to_fixed_q15(e[2 * i + 1]))) << 16);
            sink_digest(d, sink);
            continue;
        }

        header_hash(nonce, hs);

        if (V == kSha) {
            /* Same three compressions, but ⟨Z⟩ replaced by a constant so the FP64
               sweep disappears and only the SHA work remains. */
#pragma unroll
            for (int i = 0; i < QHASH_NUM_QUBITS; ++i)
                e[i] = 0.5;
        } else {
            closed_form_sweep(s_lut, NibbleWords{hs}, e);
        }

        uint32_t fw[16];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            fw[i] = hs[i];
        int zeroes = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int16_t f0 = to_fixed_q15(e[2 * i]);
            const int16_t f1 = to_fixed_q15(e[2 * i + 1]);
            const uint32_t b0 = uint32_t(f0) & 0xFFu;
            const uint32_t b1 = (uint32_t(f0) >> 8) & 0xFFu;
            const uint32_t b2 = uint32_t(f1) & 0xFFu;
            const uint32_t b3 = (uint32_t(f1) >> 8) & 0xFFu;
            fw[8 + i] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
            zeroes += int(b0 == 0) + int(b1 == 0) + int(b2 == 0) + int(b3 == 0);
        }
        uint32_t d[8];
        if (V == kProd && softfork_reject_zeroes(c_p.nTime, zeroes)) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                d[i] = 0xFFFFFFFFu;
        } else {
            sha256_init(d);
            sha256_compress(d, fw);
            sha256_compress_const(d, c_p.pad_w);
        }
        sink_digest(d, sink);
    }
}

/**
 * FP64 issue-rate probe. Sixteen independent chains keep the FP64 pipe busy, so
 * the result is throughput, not latency. kUseFma switches between the separate
 * mul+add the kernel emits today (--fmad=false) and one fused instruction.
 */
template <bool kUseFma>
__global__ void k_fp64_rate(int iters, double* sink)
{
    double x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        x[i] = double(threadIdx.x + i) * 1.0000001;
    const double a = 1.0000000001, b = 0.9999999999;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
            x[i] = kUseFma ? fma(x[i], a, b) : (x[i] * a + b);
    }
    double acc = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc += x[i];
    if (acc == 12345.6789) /* never */
        sink[0] = acc;
}

const char* name(int v)
{
    switch (v) {
    case kFull: return "full  (SHA + sweep)";
    case kSha: return "sha   (3 compressions)";
    case kSweep: return "sweep (16-step FP64)";
    case kProd: return "prod  (+softfork branch)";
    default: return "empty (loop only)";
    }
}

} // namespace

int main(int argc, char** argv)
{
    uint32_t nonces = 1u << 26;
    int threads = 256;
    int blocks = 0;
    int reps = 5;
    /* FP64 instruction count per nonce, counted from the kernel PTX with
         nvcc -ptx ... | grep -cE '\.rn\.f64'
       over the qhash_cf_kernel body. Used only for the roofline percentage. */
    int fp64_ops = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--fp64-ops" && i + 1 < argc)
            fp64_ops = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--nonces" && i + 1 < argc)
            nonces = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--threads" && i + 1 < argc)
            threads = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--blocks" && i + 1 < argc)
            blocks = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--reps" && i + 1 < argc)
            reps = int(std::strtol(argv[++i], nullptr, 10));
    }

    uint8_t header[QHASH_INPUT_SIZE];
    std::memset(header, 0xA5, sizeof(header));
    const uint32_t nTime = QHASH_SF_ANGLE;
    header[68] = uint8_t(nTime);
    header[69] = uint8_t(nTime >> 8);
    header[70] = uint8_t(nTime >> 16);
    header[71] = uint8_t(nTime >> 24);

    Params p{};
    sha256_midstate(header, p.midstate);
    p.hdr_w[0] = sha256_load_be(header + 64);
    p.hdr_w[1] = sha256_load_be(header + 68);
    p.hdr_w[2] = sha256_load_be(header + 72);
    sha256_pad_schedule(p.pad_w, QHASH_SHA256_SIZE + QHASH_NUM_QUBITS * sizeof(int16_t));
    p.lut = host_angle_lut(angle_offset(nTime));
    p.nTime = nTime;
    CHECK(cudaMemcpyToSymbol(c_p, &p, sizeof(p)));

    uint32_t* sink = nullptr;
    CHECK(cudaMalloc(&sink, sizeof(uint32_t)));
    CHECK(cudaMemset(sink, 0, sizeof(uint32_t)));

    if (blocks <= 0) {
        int sms = 0, dev = 0;
        CHECK(cudaGetDevice(&dev));
        CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        int per_sm = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_ablate<kFull>, threads, 0));
        blocks = sms * per_sm * QHASH_CF_WAVES;
        std::printf("device: %d SMs, %d blocks/SM at %d threads -> %d blocks\n", sms, per_sm,
                    threads, blocks);
    }

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    std::printf("\n%-24s %12s %12s\n", "variant", "Mh/s", "ns/nonce");
    double mhs[kVariants] = {0};
    for (int v = 0; v < kVariants; ++v) {
        double best = 0;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaEventRecord(t0));
            switch (v) {
            case kFull: k_ablate<kFull><<<blocks, threads>>>(0, nonces, sink); break;
            case kSha: k_ablate<kSha><<<blocks, threads>>>(0, nonces, sink); break;
            case kSweep: k_ablate<kSweep><<<blocks, threads>>>(0, nonces, sink); break;
            case kProd: k_ablate<kProd><<<blocks, threads>>>(0, nonces, sink); break;
            default: k_ablate<kEmpty><<<blocks, threads>>>(0, nonces, sink); break;
            }
            CHECK(cudaGetLastError());
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms = 0;
            CHECK(cudaEventElapsedTime(&ms, t0, t1));
            const double rate = double(nonces) / (double(ms) * 1e-3) / 1e6;
            if (rate > best)
                best = rate;
        }
        mhs[v] = best;
        std::printf("%-24s %12.2f %12.3f\n", name(v), best, 1e3 / best);
    }

    /* The shipped kernel, measured in this same process so it sees the same clocks
       as the ablation variants above. */
    {
        qhash_job_t job{};
        std::memcpy(job.header, header, QHASH_INPUT_SIZE);
        job.nTime = nTime;
        job.precision = QHASH_PRECISION_FP64;
        job.sim = QHASH_SIM_CLOSED_FORM;
        job.check_target = 1; /* unreachable target: keeps the share queue empty */
        job.threads_per_block = threads;
        job.blocks = blocks;
        std::memset(job.target, 0x00, 32);
        job.nonce_start = 0;
        job.nonce_count = nonces;

        qhash_share_t share{};
        double best = 0;
        for (int r = 0; r < reps; ++r) {
            uint32_t got = 0;
            double secs = 0;
            if (qhash_mine_batch(&job, &share, 1, &got, &secs) != 0) {
                std::fprintf(stderr, "qhash_mine_batch failed\n");
                return 1;
            }
            const double rate = double(nonces) / secs / 1e6;
            if (rate > best)
                best = rate;
        }
        std::printf("%-24s %12.2f %12.3f\n", "SHIPPED qhash_cf_kernel", best, 1e3 / best);
    }

    /* Serial model: if the halves did not overlap, full would cost the sum. */
    const double t_full = 1.0 / mhs[kFull];
    const double t_sha = 1.0 / mhs[kSha] - 1.0 / mhs[kEmpty];
    const double t_sweep = 1.0 / mhs[kSweep] - 1.0 / mhs[kEmpty];
    const double t_serial = 1.0 / mhs[kEmpty] + t_sha + t_sweep;
    std::printf("\nper-nonce cost share (ns, empty-loop subtracted):\n");
    std::printf("  SHA-256 half : %.3f\n", t_sha * 1e3);
    std::printf("  FP64 sweep   : %.3f\n", t_sweep * 1e3);
    std::printf("  serial model : %.3f  vs measured full %.3f (%.2fx overlap)\n", t_serial * 1e3,
                t_full * 1e3, t_serial / t_full);
    std::printf("\nverdict: %s\n",
                t_sweep > t_sha ? "FP64-sweep dominated (6.14 float-float is on the table)"
                                : "SHA-256 dominated (FP64 is not the bottleneck; skip 6.14)");

    /* ---- what the card can actually issue, so the sweep has a roofline ---- */
    {
        double* dsink = nullptr;
        CHECK(cudaMalloc(&dsink, sizeof(double)));
        const int iters = 4096;
        /* 16 chains x iters, each step either mul+add (2 ops) or one fma (2 flops). */
        const double ops_per_thread = 16.0 * double(iters) * 2.0;
        const double threads_total = double(blocks) * double(threads);

        double g_madd = 0, g_fma = 0;
        for (int fused = 0; fused < 2; ++fused) {
            double best = 0;
            for (int r = 0; r < reps; ++r) {
                CHECK(cudaEventRecord(t0));
                if (fused)
                    k_fp64_rate<true><<<blocks, threads>>>(iters, dsink);
                else
                    k_fp64_rate<false><<<blocks, threads>>>(iters, dsink);
                CHECK(cudaGetLastError());
                CHECK(cudaEventRecord(t1));
                CHECK(cudaEventSynchronize(t1));
                float ms = 0;
                CHECK(cudaEventElapsedTime(&ms, t0, t1));
                const double g = ops_per_thread * threads_total / (double(ms) * 1e-3) / 1e9;
                if (g > best)
                    best = g;
            }
            if (fused)
                g_fma = best;
            else
                g_madd = best;
        }

        /* The non-fused probe issues one instruction per counted op, so g_madd *is*
           the FP64 instruction issue rate. The fused probe retires twice the flops
           at that same instruction rate, which is the whole point of using fma. */
        const double issue = g_madd;
        std::printf("\nFP64 roofline on this device (%d blocks x %d threads):\n", blocks, threads);
        std::printf("  separate mul+add    : %8.1f Gop/s\n", g_madd);
        std::printf("  fused fma           : %8.1f Gflop/s (%.2fx the flops, same "
                    "instruction rate)\n",
                    g_fma, g_madd > 0 ? g_fma / g_madd : 0.0);
        std::printf("  FP64 instruction rate: %7.1f G/s\n", issue);
        if (fp64_ops > 0) {
            const double used = mhs[kFull] * double(fp64_ops) / 1e3;
            std::printf("  kernel issues       : %8.1f G/s at %.1f Mh/s "
                        "(%d FP64 instructions/nonce = %.0f%% of the rate)\n",
                        used, mhs[kFull], fp64_ops, issue > 0 ? 100.0 * used / issue : 0.0);
        } else {
            std::printf("  (pass --fp64-ops N, counted from the kernel PTX, to compare)\n");
        }
        cudaFree(dsink);
    }

    cudaFree(sink);
    return 0;
}
