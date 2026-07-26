/**
 * Phase 6.2 / 6.3: correctness gate for the closed-form ⟨Z⟩ sweep.
 *
 * The closed form is an exact identity, so the only way it can change a digest
 * is if its ~1e-14 floating-point residual against the FP64 statevector lands on
 * a Q1.15 rounding boundary. Every mode below attacks that question:
 *
 *   scale      1M+ real nonces, both angle offsets, closed form vs statevector.
 *              Reports the smallest rounding-boundary margin ever observed, which
 *              is the quantity that actually has to stay above the residual.
 *   boundary   Importance sampling: scan a huge number of nibble sets with the
 *              cheap closed form, keep only those whose ⟨Z⟩ sits closest to a
 *              Q1.15 boundary, and run the expensive oracle on just those.
 *   degenerate Exhaustive uniform-nibble patterns plus small-alphabet patterns —
 *              the cases that produce ⟨Z⟩ of exactly ±1 and 0, including the
 *              deliberate +32768 → −32768 int16 wrap in to_fixed_q15.
 *   softfork   Drives the zero-byte rejection paths (all four activation times)
 *              through the closed form and compares full digests to the oracle.
 *   gpu        Phase 6.5: the closed-form CUDA kernel against the CPU, digest for
 *              digest, with a strided statevector cross-check.
 *
 * Usage:
 *   qhash-closed-form-check [--mode all|scale|boundary|degenerate|softfork|gpu]
 *                           [--nonces N] [--threads T] [--scan M] [--keep K]
 */
#include "circuit.cuh"
#include "closed_form.cuh"
#include "qhash_cpu.h"
#include "qhash_kernel.cuh"
#include "qhash_params.h"
#include "reference_ld.h"
#include "sha256.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace qhash;

namespace {

constexpr int kQ = QHASH_NUM_QUBITS;
constexpr double kQ15Mult = double(1 << QHASH_FRACTION_BITS);

std::atomic<int> g_fail{0};
std::mutex g_fail_mu;

void fail(const char* fmt, ...)
{
    std::lock_guard<std::mutex> lock(g_fail_mu);
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "FAIL: ");
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
    ++g_fail;
}

/** Thread-safe informational line (diagnostics, not failures). */
void note(const char* fmt, ...)
{
    std::lock_guard<std::mutex> lock(g_fail_mu);
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::printf("\n");
    std::fflush(stdout);
}

/**
 * Distance (in Q1.15 units) from x to the nearest value that would round to a
 * different int16. to_fixed_q15 truncates x*mult ± 0.5, so the decision boundary
 * is where x*mult ± 0.5 crosses an integer.
 */
double q15_boundary_margin(double x)
{
    const double y = (x >= 0.0) ? (x * kQ15Mult + 0.5) : (x * kQ15Mult - 0.5);
    const double f = y - std::floor(y);
    return std::min(f, 1.0 - f);
}

void statevector_expect(const uint8_t nibbles[QHASH_NIBBLE_COUNT], uint32_t nTime, double out[kQ])
{
    qhash_simulate_cpu_sim(nibbles, out, QHASH_PRECISION_FP64, nTime, QHASH_SIM_STATEVECTOR);
}

void closed_form_expect(const uint8_t nibbles[QHASH_NIBBLE_COUNT], uint32_t nTime, double out[kQ])
{
    qhash_simulate_cpu_sim(nibbles, out, QHASH_PRECISION_FP64, nTime, QHASH_SIM_CLOSED_FORM);
}

/* nTime values that select each angle offset, for offset-parameterised tests. */
constexpr uint32_t kNTimeOffset0 = QHASH_SF_ANGLE - 1u;
constexpr uint32_t kNTimeOffset1 = QHASH_SF_ANGLE;

uint32_t ntime_for_offset(int offset) { return offset ? kNTimeOffset1 : kNTimeOffset0; }

void header_for_nonce(uint8_t header[QHASH_INPUT_SIZE], uint32_t nonce, uint32_t nTime)
{
    std::memset(header, 0xA5, QHASH_INPUT_SIZE);
    header[68] = uint8_t(nTime);
    header[69] = uint8_t(nTime >> 8);
    header[70] = uint8_t(nTime >> 16);
    header[71] = uint8_t(nTime >> 24);
    header[76] = uint8_t(nonce);
    header[77] = uint8_t(nonce >> 8);
    header[78] = uint8_t(nonce >> 16);
    header[79] = uint8_t(nonce >> 24);
}

void nibbles_for_nonce(uint8_t nibbles[QHASH_NIBBLE_COUNT], uint32_t nonce, uint32_t nTime)
{
    uint8_t header[QHASH_INPUT_SIZE];
    header_for_nonce(header, nonce, nTime);
    uint8_t hash[QHASH_SHA256_SIZE];
    sha256(header, QHASH_INPUT_SIZE, hash);
    split_nibbles(hash, nibbles);
}

/* ---------- shared comparison accumulator ---------- */

struct Stats {
    uint64_t qubit_checks = 0;
    uint64_t q15_mismatch = 0;
    uint64_t q15_mismatch_tie = 0; /* mismatches whose exact value is a boundary tie */
    uint64_t exact_ties = 0;       /* values that are exact ties, agreed or not */
    uint64_t tie_cases = 0;        /* nibble sets containing at least one exact tie */
    uint64_t case_mismatch = 0;
    uint64_t cases = 0;
    double max_abs_diff = 0.0;
    /* Tightest boundary margin among values that are NOT exact ties — this is the
       quantity the closed form's ~1e-14 residual actually has to stay below. */
    long double min_margin_resolvable = 1e30L;
    double min_margin_diff = 0.0; /* |Δ⟨Z⟩| in Q1.15 units at that point */

    void merge(const Stats& o)
    {
        qubit_checks += o.qubit_checks;
        q15_mismatch += o.q15_mismatch;
        q15_mismatch_tie += o.q15_mismatch_tie;
        exact_ties += o.exact_ties;
        tie_cases += o.tie_cases;
        case_mismatch += o.case_mismatch;
        cases += o.cases;
        max_abs_diff = std::max(max_abs_diff, o.max_abs_diff);
        if (o.min_margin_resolvable < min_margin_resolvable) {
            min_margin_resolvable = o.min_margin_resolvable;
            min_margin_diff = o.min_margin_diff;
        }
    }
};

/** Compare one nibble set; returns true when all 16 Q1.15 values agree. */
bool compare_case(const uint8_t nibbles[QHASH_NIBBLE_COUNT], uint32_t nTime, Stats& st,
                  const char* tag, uint64_t index)
{
    double ref[kQ], got[kQ];
    statevector_expect(nibbles, nTime, ref);
    closed_form_expect(nibbles, nTime, got);

    /* One long-double sweep per case gives the exact value of all 16 ⟨Z⟩, which is
       what separates "this input has no FP64 answer" from "we got it wrong". */
    long double exact[kQ];
    qhash_test::reference_sweep_ld(nibbles, angle_offset(nTime), exact);

    bool ok = true;
    bool any_tie = false;
    ++st.cases;
    for (int q = 0; q < kQ; ++q) {
        ++st.qubit_checks;
        const double d = std::fabs(ref[q] - got[q]);
        st.max_abs_diff = std::max(st.max_abs_diff, d);

        const long double exact_margin = qhash_test::q15_boundary_margin_ld(exact[q]);
        const bool tie = exact_margin < qhash_test::kQ15TieThreshold;
        if (tie) {
            ++st.exact_ties;
            any_tie = true;
        } else if (exact_margin < st.min_margin_resolvable) {
            st.min_margin_resolvable = exact_margin;
            st.min_margin_diff = d * kQ15Mult;
        }

        const int16_t fr = to_fixed_q15(ref[q]);
        const int16_t fg = to_fixed_q15(got[q]);
        if (fr == fg)
            continue;

        ++st.q15_mismatch;
        if (tie)
            ++st.q15_mismatch_tie;
        if (ok)
            note("  %s case %llu qubit %d: Q1.15 %d vs %d, exact=%.21Lg, "
                 "exact boundary margin=%.3Le -> %s",
                 tag, (unsigned long long)index, q, int(fr), int(fg), exact[q], exact_margin,
                 tie ? "EXACT TIE (no FP64 answer is defined)" : "REAL DIVERGENCE");
        if (!tie)
            fail("%s case %llu qubit %d: Q1.15 %d vs %d with exact value %.21Lg off the boundary "
                 "by %.3Le — closed form diverges", tag, (unsigned long long)index, q, int(fr),
                 int(fg), exact[q], exact_margin);
        ok = false;
    }
    if (any_tie)
        ++st.tie_cases;
    if (!ok)
        ++st.case_mismatch;
    return ok;
}

void report(const char* tag, const Stats& st)
{
    const double diff_q15 = st.max_abs_diff * kQ15Mult;
    std::printf("\n=== %s ===\n", tag);
    std::printf("  cases                     : %llu\n", (unsigned long long)st.cases);
    std::printf("  Q1.15 mismatches          : %llu / %llu  (%llu on exact ties, %llu real)\n",
                (unsigned long long)st.q15_mismatch, (unsigned long long)st.qubit_checks,
                (unsigned long long)st.q15_mismatch_tie,
                (unsigned long long)(st.q15_mismatch - st.q15_mismatch_tie));
    std::printf("  case mismatches           : %llu\n", (unsigned long long)st.case_mismatch);
    std::printf("  exact Q1.15 ties in input : %llu values / %llu cases", 
                (unsigned long long)st.exact_ties, (unsigned long long)st.tie_cases);
    if (st.cases)
        std::printf("  (%.3g per case)", double(st.tie_cases) / double(st.cases));
    std::printf("\n");
    std::printf("  max |Δ⟨Z⟩|                : %.3e  (= %.3e Q1.15 units)\n", st.max_abs_diff,
                diff_q15);
    if (st.min_margin_resolvable < 1e29L) {
        std::printf("  tightest resolvable margin: %.3Le Q1.15 units (|Δ| there %.3e)\n",
                    st.min_margin_resolvable, st.min_margin_diff);
        if (diff_q15 > 0.0)
            std::printf("  margin / worst-case |Δ|   : %.1fx\n",
                        double(st.min_margin_resolvable) / diff_q15);
    }
    if (st.q15_mismatch == 0)
        std::printf("  RESULT                    : all Q1.15 values identical\n");
    else if (st.q15_mismatch == st.q15_mismatch_tie)
        std::printf("  RESULT                    : differs only on exact boundary ties\n");
    else
        std::printf("  RESULT                    : MISMATCH\n");
}

/** Run `total` work items across `nthreads` threads; body(index, thread_stats). */
template <typename Body>
Stats parallel_for(uint64_t total, int nthreads, const char* progress_tag, Body body)
{
    std::vector<Stats> per(nthreads);
    std::atomic<uint64_t> next{0};
    std::atomic<uint64_t> done{0};
    const uint64_t grain = 256;

    auto worker = [&](int t) {
        for (;;) {
            const uint64_t begin = next.fetch_add(grain);
            if (begin >= total)
                return;
            const uint64_t end = std::min(begin + grain, total);
            for (uint64_t i = begin; i < end; ++i)
                body(i, per[t]);
            const uint64_t d = done.fetch_add(end - begin) + (end - begin);
            if (progress_tag && (d % (grain * 400) < grain || d == total)) {
                std::printf("  %s %llu / %llu\n", progress_tag, (unsigned long long)d,
                            (unsigned long long)total);
                std::fflush(stdout);
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; ++t)
        pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool)
        th.join();

    Stats all;
    for (const auto& s : per)
        all.merge(s);
    return all;
}

/* ---------- mode: scale ---------- */

/**
 * Exact ties are far more common at angle offset 0, where θ is a multiple of π/16
 * and cos/sin collapse to dyadic-friendly values, than at offset 1 (odd multiples
 * of π/32). The live network has run at offset 1 since Sep 2025, so the two are
 * scored separately rather than averaged.
 */
int mode_scale(uint64_t nonces, int nthreads)
{
    int rc = 0;
    for (int offset = 0; offset < 2; ++offset) {
        const uint32_t nTime = ntime_for_offset(offset);
        std::printf("[scale] %llu nonces at angle offset %d (nTime=%u), %d threads\n",
                    (unsigned long long)nonces, offset, nTime, nthreads);
        Stats st = parallel_for(nonces, nthreads, "scale", [nTime](uint64_t i, Stats& s) {
            uint8_t nibbles[QHASH_NIBBLE_COUNT];
            nibbles_for_nonce(nibbles, uint32_t(i), nTime);
            compare_case(nibbles, nTime, s, "scale", i);
        });
        char tag[128];
        std::snprintf(tag, sizeof(tag), "scale offset %d: closed form vs FP64 statevector", offset);
        report(tag, st);
        if (st.q15_mismatch != st.q15_mismatch_tie)
            rc = 1;
    }
    return rc;
}

/* ---------- mode: boundary (importance sampling) ---------- */

struct Candidate {
    double margin;
    uint8_t nibbles[QHASH_NIBBLE_COUNT];
    int offset;
};

int mode_boundary(uint64_t scan, size_t keep, int nthreads)
{
    std::printf("[boundary] scanning %llu nibble sets with the closed form, "
                "keeping the %zu closest to a Q1.15 boundary\n",
                (unsigned long long)scan, keep);

    std::mutex mu;
    std::vector<Candidate> best;
    double worst_kept = 1e9;

    std::atomic<uint64_t> next{0};
    const uint64_t grain = 4096;

    auto worker = [&](uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::vector<Candidate> local;
        for (;;) {
            const uint64_t begin = next.fetch_add(grain);
            if (begin >= scan)
                break;
            const uint64_t end = std::min(begin + grain, scan);
            for (uint64_t i = begin; i < end; ++i) {
                Candidate c;
                c.offset = int(rng() & 1);
                for (int k = 0; k < QHASH_NIBBLE_COUNT; ++k)
                    c.nibbles[k] = uint8_t(rng() & 0xF);
                double e[kQ];
                closed_form_expectations(host_angle_lut(c.offset), c.nibbles, e);
                double m = 1.0;
                for (int q = 0; q < kQ; ++q)
                    m = std::min(m, q15_boundary_margin(e[q]));
                c.margin = m;
                if (m < worst_kept || local.size() < keep) {
                    local.push_back(c);
                    if (local.size() > keep * 4) {
                        std::sort(local.begin(), local.end(),
                                  [](const Candidate& a, const Candidate& b) {
                                      return a.margin < b.margin;
                                  });
                        local.resize(keep);
                    }
                }
            }
            std::lock_guard<std::mutex> lock(mu);
            if (!local.empty()) {
                best.insert(best.end(), local.begin(), local.end());
                local.clear();
                std::sort(best.begin(), best.end(),
                          [](const Candidate& a, const Candidate& b) { return a.margin < b.margin; });
                if (best.size() > keep)
                    best.resize(keep);
                worst_kept = best.back().margin;
            }
        }
        std::lock_guard<std::mutex> lock(mu);
        best.insert(best.end(), local.begin(), local.end());
        std::sort(best.begin(), best.end(),
                  [](const Candidate& a, const Candidate& b) { return a.margin < b.margin; });
        if (best.size() > keep)
            best.resize(keep);
    };

    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; ++t)
        pool.emplace_back(worker, 0x9E3779B97F4A7C15ull * uint64_t(t + 1));
    worker(0x9E3779B97F4A7C15ull);
    for (auto& th : pool)
        th.join();

    std::printf("  tightest margins found: ");
    for (size_t i = 0; i < best.size() && i < 8; ++i)
        std::printf("%.3e ", best[i].margin);
    std::printf("\n  now running the statevector oracle on %zu candidates\n", best.size());

    Stats st = parallel_for(best.size(), nthreads, nullptr, [&](uint64_t i, Stats& s) {
        compare_case(best[i].nibbles, ntime_for_offset(best[i].offset), s, "boundary", i);
    });
    report("boundary: tightest Q1.15 margins", st);
    return st.q15_mismatch ? 1 : 0;
}

/* ---------- mode: degenerate angles ---------- */

/** Exhaustive uniform patterns: every qubit gets the same (ny1, nz1, ny2). */
int degenerate_uniform(int nthreads)
{
    const uint64_t total = 16ull * 16ull * 16ull * 2ull;
    std::printf("[degenerate] %llu uniform (ny1,nz1,ny2) x offset patterns\n",
                (unsigned long long)total);

    std::atomic<uint64_t> exact_one{0}, exact_zero{0}, wrapped{0};

    Stats st = parallel_for(total, nthreads, nullptr, [&](uint64_t i, Stats& s) {
        const int offset = int(i & 1);
        const uint64_t p = i >> 1;
        const uint8_t ny1 = uint8_t(p & 0xF);
        const uint8_t nz1 = uint8_t((p >> 4) & 0xF);
        const uint8_t ny2 = uint8_t((p >> 8) & 0xF);

        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        for (int q = 0; q < kQ; ++q) {
            nibbles[q] = ny1;
            nibbles[kQ + q] = nz1;
            nibbles[2 * kQ + q] = ny2;
            /* Layer-2 Rz: the statevector applies it, the closed form ignores it.
               Keep it non-trivial so any residual influence would show up here. */
            nibbles[3 * kQ + q] = uint8_t((ny1 + 5 + q) & 0xF);
        }

        const uint32_t nTime = ntime_for_offset(offset);
        compare_case(nibbles, nTime, s, "uniform", i);

        double e[kQ];
        closed_form_expect(nibbles, nTime, e);
        for (int q = 0; q < kQ; ++q) {
            if (e[q] == 1.0 || e[q] == -1.0)
                ++exact_one;
            if (e[q] == 0.0)
                ++exact_zero;
            if (to_fixed_q15(e[q]) == int16_t(-32768) && e[q] > 0.0)
                ++wrapped;
        }
    });

    std::printf("  exact ±1 values: %llu   exact 0 values: %llu   +32768→−32768 wraps: %llu\n",
                (unsigned long long)exact_one.load(), (unsigned long long)exact_zero.load(),
                (unsigned long long)wrapped.load());
    if (exact_one.load() == 0)
        fail("degenerate sweep produced no exact ±1 values — test is not exercising the edge");
    if (wrapped.load() == 0)
        fail("degenerate sweep never hit the +32768 → −32768 int16 wrap");
    report("degenerate: uniform nibble patterns", st);
    return st.q15_mismatch ? 1 : 0;
}

/**
 * Random patterns over the alphabet {0,4,8,12}: θ is a multiple of π/4 (offset 0),
 * which is where cos/sin collapse to 0, ±1, ±1/√2 and ties are most likely.
 */
int degenerate_alphabet(uint64_t count, int nthreads)
{
    std::printf("[degenerate] %llu random patterns over nibbles {0,4,8,12}\n",
                (unsigned long long)count);
    Stats st = parallel_for(count, nthreads, nullptr, [](uint64_t i, Stats& s) {
        std::mt19937_64 rng(0xDEADBEEF00000000ull + i);
        static const uint8_t kAlpha[4] = {0, 4, 8, 12};
        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        for (int k = 0; k < QHASH_NIBBLE_COUNT; ++k)
            nibbles[k] = kAlpha[rng() & 3];
        const int offset = int(i & 1);
        compare_case(nibbles, ntime_for_offset(offset), s, "alphabet", i);
    });
    report("degenerate: quarter-turn alphabet", st);
    return st.q15_mismatch ? 1 : 0;
}

/** The identity circuit: offset 0 and all-zero angle nibbles ⇒ every ⟨Z⟩ = +1. */
int degenerate_identity()
{
    std::printf("[degenerate] identity circuit (offset 0, zero angle nibbles)\n");
    uint8_t nibbles[QHASH_NIBBLE_COUNT];
    std::memset(nibbles, 0, sizeof(nibbles));
    for (int i = 3 * kQ; i < QHASH_NIBBLE_COUNT; ++i)
        nibbles[i] = uint8_t(i & 0xF); /* dead layer-2 Rz inputs must not matter */

    double ref[kQ], got[kQ];
    statevector_expect(nibbles, kNTimeOffset0, ref);
    closed_form_expect(nibbles, kNTimeOffset0, got);

    for (int q = 0; q < kQ; ++q) {
        if (got[q] != 1.0)
            fail("identity circuit: closed form ⟨Z_%d⟩ = %.17g, expected exactly 1.0", q, got[q]);
        if (to_fixed_q15(ref[q]) != to_fixed_q15(got[q]))
            fail("identity circuit: Q1.15 %d (statevector %.17g) != %d (closed form %.17g)",
                 int(to_fixed_q15(ref[q])), ref[q], int(to_fixed_q15(got[q])), got[q]);
        if (to_fixed_q15(got[q]) != int16_t(-32768))
            fail("identity circuit: Q1.15(1.0) = %d, expected the −32768 wrap",
                 int(to_fixed_q15(got[q])));
    }
    /* The statevector overshoots 1.0 slightly because the layer-2 Rz phases make
       |amp|² = cos²+sin² ≠ 1 in FP64; both still land on the same int16. */
    std::printf("  closed form gives exactly +1.0 (statevector %.17g); Q1.15 wraps to −32768\n",
                ref[0]);
    return 0;
}

/** to_fixed_q15 must not change: check the wrap and both sides of a boundary. */
int fixed_point_edges()
{
    std::printf("[degenerate] to_fixed_q15 edges\n");
    struct Case {
        double x;
        int16_t want;
        const char* what;
    };
    const double eps = 1e-15;
    const Case cases[] = {
        {1.0, int16_t(32768), "+1.0 wraps to −32768"},
        {-1.0, int16_t(-32768), "−1.0"},
        {std::nextafter(1.0, 0.0), int16_t(32768), "just below +1.0 still wraps"},
        {0.0, 0, "zero"},
        {-0.0, 0, "negative zero"},
        {0.5, 16384, "0.5"},
        {(0.5 + eps) / kQ15Mult, 1, "just above the 0→1 boundary"},
        {(0.5 - eps) / kQ15Mult, 0, "just below the 0→1 boundary"},
        {-(0.5 + eps) / kQ15Mult, -1, "just below the 0→−1 boundary"},
        {-(0.5 - eps) / kQ15Mult, 0, "just above the 0→−1 boundary"},
    };
    for (const Case& c : cases) {
        const int16_t got = to_fixed_q15(c.x);
        if (got != c.want)
            fail("to_fixed_q15(%.17g) = %d, expected %d (%s)", c.x, int(got), int(c.want), c.what);
    }
    std::printf("  %zu edge cases OK (int16 wrap preserved)\n", sizeof(cases) / sizeof(cases[0]));
    return 0;
}

/* ---------- mode: soft-fork zero-byte rejection ---------- */

int count_zero_bytes(const double e[kQ])
{
    int zeroes = 0;
    for (int q = 0; q < kQ; ++q) {
        const int16_t f = to_fixed_q15(e[q]);
        if (uint8_t(f & 0xFF) == 0)
            ++zeroes;
        if (uint8_t((f >> 8) & 0xFF) == 0)
            ++zeroes;
    }
    return zeroes;
}

int mode_softfork(int nthreads)
{
    std::printf("[softfork] searching uniform patterns for high zero-byte counts\n");

    /* Find, per zero-byte count, one uniform nibble pattern that produces it. */
    struct Best {
        int zeroes = -1;
        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        int offset = 0;
    };
    std::vector<Best> by_count(33);
    std::mutex mu;

    const uint64_t total = 16ull * 16ull * 16ull * 2ull;
    parallel_for(total, nthreads, nullptr, [&](uint64_t i, Stats&) {
        const int offset = int(i & 1);
        const uint64_t p = i >> 1;
        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        for (int q = 0; q < kQ; ++q) {
            nibbles[q] = uint8_t(p & 0xF);
            nibbles[kQ + q] = uint8_t((p >> 4) & 0xF);
            nibbles[2 * kQ + q] = uint8_t((p >> 8) & 0xF);
            nibbles[3 * kQ + q] = 0;
        }
        double e[kQ];
        closed_form_expectations(host_angle_lut(offset), nibbles, e);
        const int z = count_zero_bytes(e);
        std::lock_guard<std::mutex> lock(mu);
        if (by_count[z].zeroes < 0) {
            by_count[z].zeroes = z;
            by_count[z].offset = offset;
            std::memcpy(by_count[z].nibbles, nibbles, QHASH_NIBBLE_COUNT);
        }
    });

    std::printf("  zero-byte counts reachable from uniform patterns:");
    for (int z = 0; z <= 32; ++z)
        if (by_count[z].zeroes >= 0)
            std::printf(" %d", z);
    std::printf("\n");

    /* Every soft-fork activation time, plus one second before each. */
    const uint32_t times[] = {0,
                              QHASH_SF_ZERO_ALL - 1u, QHASH_SF_ZERO_ALL,
                              QHASH_SF_ZERO_3_4 - 1u, QHASH_SF_ZERO_3_4,
                              QHASH_SF_ZERO_1_4 - 1u, QHASH_SF_ZERO_1_4,
                              QHASH_SF_ANGLE - 1u,    QHASH_SF_ANGLE};

    uint64_t compared = 0, rejected = 0, digest_mismatch = 0;
    uint8_t in_hash[QHASH_SHA256_SIZE];
    for (int i = 0; i < QHASH_SHA256_SIZE; ++i)
        in_hash[i] = uint8_t(0x11 * i + 7);

    for (int z = 0; z <= 32; ++z) {
        if (by_count[z].zeroes < 0)
            continue;
        for (uint32_t t : times) {
            /* Only compare where the pattern's angle offset matches this nTime,
               otherwise we would be simulating a different circuit. */
            if (angle_offset(t) != by_count[z].offset)
                continue;

            double sv[kQ], cf[kQ];
            statevector_expect(by_count[z].nibbles, t, sv);
            closed_form_expect(by_count[z].nibbles, t, cf);

            uint8_t d_sv[QHASH_SHA256_SIZE], d_cf[QHASH_SHA256_SIZE];
            qhash_finalize_cpu(in_hash, sv, t, d_sv);
            qhash_finalize_cpu(in_hash, cf, t, d_cf);
            ++compared;

            const bool reject = softfork_reject_zeroes(t, count_zero_bytes(cf));
            if (reject)
                ++rejected;
            if (std::memcmp(d_sv, d_cf, QHASH_SHA256_SIZE) != 0) {
                ++digest_mismatch;
                fail("softfork digest differs: zeroes=%d nTime=%u", z, t);
            }
            if (reject) {
                for (int b = 0; b < QHASH_SHA256_SIZE; ++b)
                    if (d_cf[b] != 0xFF) {
                        fail("softfork rejection did not produce the 0xFF digest "
                             "(zeroes=%d nTime=%u)", z, t);
                        break;
                    }
            }
        }
    }

    std::printf("\n=== softfork: zero-byte rejection through the closed form ===\n");
    std::printf("  digests compared   : %llu\n", (unsigned long long)compared);
    std::printf("  0xFF rejections hit: %llu\n", (unsigned long long)rejected);
    std::printf("  digest mismatches  : %llu\n", (unsigned long long)digest_mismatch);
    if (rejected == 0)
        fail("no soft-fork rejection path was exercised");

    /* Also confirm the closed form counts the same zero bytes as the oracle on
       ordinary nonces, where the count is small but nonzero. */
    constexpr uint64_t kZeroByteNonces = 20000;
    std::atomic<uint64_t> zero_byte_disagreements{0};
    parallel_for(kZeroByteNonces, nthreads, nullptr, [&](uint64_t i, Stats&) {
        const uint32_t n = uint32_t(i);
        const uint32_t nt = ntime_for_offset(int(n & 1));
        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        nibbles_for_nonce(nibbles, n, nt);
        double sv[kQ], cf[kQ];
        statevector_expect(nibbles, nt, sv);
        closed_form_expect(nibbles, nt, cf);
        if (count_zero_bytes(sv) != count_zero_bytes(cf))
            ++zero_byte_disagreements;
    });
    std::printf("  zero-byte count disagreements on %llu real nonces: %llu\n",
                (unsigned long long)kZeroByteNonces,
                (unsigned long long)zero_byte_disagreements.load());
    if (zero_byte_disagreements.load())
        fail("closed form and statevector disagree on the soft-fork zero-byte count");

    return digest_mismatch ? 1 : 0;
}

/** Full-digest equivalence through qhash_hash_cpu_sim on real headers. */
int mode_digest(uint64_t nonces, int nthreads)
{
    std::printf("[digest] %llu full qhash digests, closed form vs statevector\n",
                (unsigned long long)nonces);
    std::atomic<uint64_t> mismatch{0};
    parallel_for(nonces, nthreads, nullptr, [&](uint64_t i, Stats&) {
        const uint32_t nTime = ntime_for_offset(int(i & 1));
        uint8_t header[QHASH_INPUT_SIZE];
        header_for_nonce(header, uint32_t(i), nTime);
        uint8_t a[QHASH_SHA256_SIZE], b[QHASH_SHA256_SIZE];
        qhash_hash_cpu_sim(header, a, QHASH_PRECISION_FP64, nTime, QHASH_SIM_STATEVECTOR);
        qhash_hash_cpu_sim(header, b, QHASH_PRECISION_FP64, nTime, QHASH_SIM_CLOSED_FORM);
        if (std::memcmp(a, b, QHASH_SHA256_SIZE) != 0)
            ++mismatch;
    });
    std::printf("  digest mismatches: %llu / %llu\n", (unsigned long long)mismatch.load(),
                (unsigned long long)nonces);
    if (mismatch.load())
        fail("full digests differ between closed form and statevector");
    return mismatch.load() ? 1 : 0;
}

/**
 * Arbitrate a single nonce. When a wide sweep reports a digest disagreement it
 * only reports the nonce, so this reruns that one case with the long-double
 * reference and prints all 16 ⟨Z⟩ side by side — enough to tell an exact boundary
 * tie from a genuine divergence without rerunning the sweep.
 */
int mode_nonce(uint32_t nonce, int offset)
{
    const uint32_t nTime = ntime_for_offset(offset);
    std::printf("[nonce] %u, angle offset %d (nTime %u)\n", nonce, offset, nTime);

    uint8_t nibbles[QHASH_NIBBLE_COUNT];
    nibbles_for_nonce(nibbles, nonce, nTime);

    double ref[kQ], got[kQ];
    statevector_expect(nibbles, nTime, ref);
    closed_form_expect(nibbles, nTime, got);
    long double exact[kQ];
    qhash_test::reference_sweep_ld(nibbles, angle_offset(nTime), exact);

    std::printf("   q  nib(y1,z1,y2)  Q1.15 sv/cf   statevector          closed form"
                "          exact (long double)   margin     verdict\n");
    for (int q = 0; q < kQ; ++q) {
        const int16_t fr = to_fixed_q15(ref[q]);
        const int16_t fg = to_fixed_q15(got[q]);
        const long double margin = qhash_test::q15_boundary_margin_ld(exact[q]);
        const bool tie = margin < qhash_test::kQ15TieThreshold;
        const char* verdict = (fr == fg) ? "agree" : (tie ? "EXACT TIE" : "REAL DIVERGENCE");
        std::printf("  %2d  (%2u,%2u,%2u)   %6d/%-6d %.17g  %.17g  %.21Lg  %.3Le  %s\n", q,
                    unsigned(nibbles[q]), unsigned(nibbles[16 + q]), unsigned(nibbles[32 + q]),
                    int(fr), int(fg), ref[q], got[q], exact[q], margin, verdict);
    }

    /* Which of the two agrees with the exact value decides whether the closed
       form is the one that is wrong. */
    Stats st;
    const bool ok = compare_case(nibbles, nTime, st, "nonce", nonce);
    for (int q = 0; q < kQ; ++q) {
        if (to_fixed_q15(ref[q]) == to_fixed_q15(got[q]))
            continue;
        const int16_t fe = to_fixed_q15(double(exact[q]));
        std::printf("  qubit %d: long-double rounds to %d -> statevector is %s, closed form is %s\n",
                    q, int(fe), to_fixed_q15(ref[q]) == fe ? "right" : "wrong",
                    to_fixed_q15(got[q]) == fe ? "right" : "wrong");
    }
    std::printf("  digest-affecting mismatch : %s\n", ok ? "no" : "yes");
    return ok ? 0 : 1;
}

/**
 * Phase 6.5 gate: the closed-form CUDA kernel must reproduce the CPU digest for
 * every nonce. Each chunk is compared against the CPU closed form (cheap enough
 * to cover millions of nonces) and, on a strided sample, against the statevector
 * oracle as well, so the chain GPU → CPU closed form → statevector is closed.
 *
 * GPU vs CPU closed form must be exact — same arithmetic, so any difference is a
 * defect. GPU vs oracle is arbitrated like every other mode: a difference on an
 * exact Q1.15 tie is expected at a rate of roughly one nonce in 10^5 and is not a
 * defect, because no FP64 evaluation order has a defined answer there.
 */
int mode_gpu(uint64_t nonces, int nthreads, uint32_t chunk, uint32_t oracle_stride,
             int threads_per_block, int blocks)
{
    if (!qhash_cuda_available()) {
        std::printf("[gpu] no CUDA device — skipping\n");
        return 0;
    }
    std::printf("[gpu] %llu nonces, closed-form kernel vs CPU (chunk=%u, threads=%d, blocks=%d, "
                "oracle every %u)\n",
                (unsigned long long)nonces, chunk, threads_per_block, blocks, oracle_stride);

    std::vector<qhash_share_t> shares(chunk);
    std::vector<uint8_t> seen(chunk);
    std::atomic<uint64_t> cpu_mismatch{0}, oracle_mismatch{0}, oracle_checked{0}, oracle_tie{0};
    double gpu_seconds = 0;

    for (uint64_t base = 0; base < nonces; base += chunk) {
        const uint32_t n = uint32_t(std::min<uint64_t>(chunk, nonces - base));
        /* Alternate the angle offset between chunks so both LUTs get exercised. */
        const uint32_t nTime = ntime_for_offset(int((base / chunk) & 1));

        qhash_job_t job{};
        header_for_nonce(job.header, 0, nTime);
        job.nonce_start = uint32_t(base);
        job.nonce_count = n;
        job.nTime = nTime;
        job.precision = QHASH_PRECISION_FP64;
        job.sim = QHASH_SIM_CLOSED_FORM;
        job.check_target = 0; /* record every nonce */
        job.threads_per_block = threads_per_block;
        job.blocks = blocks;
        std::memset(job.target, 0xFF, 32);

        uint32_t got = 0;
        double secs = 0;
        if (qhash_mine_batch(&job, shares.data(), n, &got, &secs) != 0) {
            fail("gpu: mine_batch failed at nonce %llu", (unsigned long long)base);
            return 1;
        }
        gpu_seconds += secs;
        if (got != n) {
            fail("gpu: batch at %llu returned %u shares, expected %u",
                 (unsigned long long)base, got, n);
            return 1;
        }

        /* The share queue order is nondeterministic; index it by nonce. */
        std::fill(seen.begin(), seen.begin() + n, 0);
        std::vector<const qhash_share_t*> by_nonce(n, nullptr);
        for (uint32_t i = 0; i < n; ++i) {
            const uint64_t off = uint64_t(shares[i].nonce) - base;
            if (off >= n || seen[off]) {
                fail("gpu: share queue has bad or duplicate nonce %u", shares[i].nonce);
                return 1;
            }
            seen[off] = 1;
            by_nonce[off] = &shares[i];
        }

        parallel_for(n, nthreads, nullptr, [&](uint64_t i, Stats&) {
            const uint32_t nonce = uint32_t(base + i);
            uint8_t header[QHASH_INPUT_SIZE];
            header_for_nonce(header, nonce, nTime);

            uint8_t cpu[QHASH_SHA256_SIZE];
            qhash_hash_cpu_sim(header, cpu, QHASH_PRECISION_FP64, nTime, QHASH_SIM_CLOSED_FORM);
            if (std::memcmp(by_nonce[i]->hash, cpu, QHASH_SHA256_SIZE) != 0) {
                ++cpu_mismatch;
                fail("gpu: nonce %u digest differs from CPU closed form", nonce);
            }
            if (oracle_stride && (nonce % oracle_stride) == 0) {
                uint8_t sv[QHASH_SHA256_SIZE];
                qhash_hash_cpu_sim(header, sv, QHASH_PRECISION_FP64, nTime,
                                   QHASH_SIM_STATEVECTOR);
                ++oracle_checked;
                if (std::memcmp(by_nonce[i]->hash, sv, QHASH_SHA256_SIZE) != 0) {
                    /* A digest difference against the oracle is only a defect if some
                       qubit is off a rounding boundary; on an exact tie neither FP64
                       answer is defined. compare_case arbitrates with the long-double
                       reference and fails only on a real divergence. */
                    Stats one;
                    uint8_t nibbles[QHASH_NIBBLE_COUNT];
                    nibbles_for_nonce(nibbles, nonce, nTime);
                    compare_case(nibbles, nTime, one, "gpu-oracle", nonce);
                    if (one.q15_mismatch > one.q15_mismatch_tie)
                        ++oracle_mismatch;
                    else
                        ++oracle_tie;
                }
            }
        });

        if (((base / chunk) & 15) == 0 || base + n >= nonces) {
            note("  gpu progress %llu / %llu  cpu-mismatch=%llu oracle-real=%llu "
                 "oracle-tie=%llu / %llu checked",
                 (unsigned long long)(base + n), (unsigned long long)nonces,
                 (unsigned long long)cpu_mismatch.load(),
                 (unsigned long long)oracle_mismatch.load(), (unsigned long long)oracle_tie.load(),
                 (unsigned long long)oracle_checked.load());
        }
    }

    std::printf("\n=== gpu: closed-form kernel vs CPU ===\n");
    std::printf("  nonces compared            : %llu\n", (unsigned long long)nonces);
    std::printf("  vs CPU closed form         : %llu mismatches\n",
                (unsigned long long)cpu_mismatch.load());
    std::printf("  vs statevector oracle      : %llu real divergences, %llu exact ties / "
                "%llu checked\n",
                (unsigned long long)oracle_mismatch.load(), (unsigned long long)oracle_tie.load(),
                (unsigned long long)oracle_checked.load());
    std::printf("  kernel time                : %.3f s (%.3f Mh/s)\n", gpu_seconds,
                gpu_seconds > 0 ? double(nonces) / gpu_seconds / 1e6 : 0.0);
    return (cpu_mismatch.load() || oracle_mismatch.load()) ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string mode = "all";
    uint64_t nonces = 1000000;
    uint64_t scan = 20000000;
    size_t keep = 512;
    uint32_t gpu_chunk = 1u << 20;
    uint32_t oracle_stride = 4096;
    int gpu_threads = 0;
    int gpu_blocks = 0;
    uint32_t one_nonce = 0;
    int one_offset = 0;
    int nthreads = int(std::thread::hardware_concurrency());
    if (nthreads < 1)
        nthreads = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mode" && i + 1 < argc)
            mode = argv[++i];
        else if (a == "--nonces" && i + 1 < argc)
            nonces = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--scan" && i + 1 < argc)
            scan = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--keep" && i + 1 < argc)
            keep = size_t(std::strtoull(argv[++i], nullptr, 10));
        else if (a == "--threads" && i + 1 < argc)
            nthreads = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--gpu-chunk" && i + 1 < argc)
            gpu_chunk = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--oracle-stride" && i + 1 < argc)
            oracle_stride = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--gpu-threads" && i + 1 < argc)
            gpu_threads = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--gpu-blocks" && i + 1 < argc)
            gpu_blocks = int(std::strtol(argv[++i], nullptr, 10));
        else if (a == "--nonce" && i + 1 < argc)
            one_nonce = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--offset" && i + 1 < argc)
            one_offset = int(std::strtol(argv[++i], nullptr, 10)) ? 1 : 0;
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--mode all|scale|boundary|degenerate|softfork|digest|gpu|nonce]\n"
                        "          [--nonces N] [--scan M] [--keep K] [--threads T]\n"
                        "          [--gpu-chunk C] [--oracle-stride S] [--gpu-threads T]\n"
                        "          [--gpu-blocks B] [--nonce N --offset 0|1]\n",
                        argv[0]);
            return 0;
        }
    }

    std::printf("closed-form check: mode=%s threads=%d\n", mode.c_str(), nthreads);

    const bool all = (mode == "all");
    if (mode == "nonce") {
        mode_nonce(one_nonce, one_offset);
        if (g_fail) {
            std::fprintf(stderr, "\nclosed-form check FAILED (%d problem(s))\n", g_fail.load());
            return 1;
        }
        std::printf("\nclosed-form check PASSED\n");
        return 0;
    }
    if (all || mode == "degenerate") {
        fixed_point_edges();
        degenerate_identity();
        degenerate_uniform(nthreads);
        degenerate_alphabet(20000, nthreads);
    }
    if (all || mode == "softfork")
        mode_softfork(nthreads);
    if (all || mode == "boundary")
        mode_boundary(scan, keep, nthreads);
    if (all || mode == "digest")
        mode_digest(std::min<uint64_t>(nonces, 100000), nthreads);
    if (all || mode == "scale")
        mode_scale(nonces, nthreads);
    if (all || mode == "gpu")
        mode_gpu(nonces, nthreads, gpu_chunk == 0 ? 1u : gpu_chunk, oracle_stride, gpu_threads,
                 gpu_blocks);

    if (g_fail) {
        std::fprintf(stderr, "\nclosed-form check FAILED (%d problem(s))\n", g_fail.load());
        return 1;
    }
    std::printf("\nclosed-form check PASSED\n");
    return 0;
}
