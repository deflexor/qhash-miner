/**
 * Prototype: closed-form ⟨Z⟩ for the qhash circuit without a 65536-amplitude
 * statevector.
 *
 * Claim under test: because each cut (k,k+1) is crossed by exactly two CNOTs and
 * the final CNOT staircase maps Z_q -> Z_0...Z_q in the Heisenberg picture, all
 * 16 expectation values follow from a single left-to-right sweep of 2x2 matrices.
 *
 * Build:
 *   g++ -O2 -Iinclude tests/mps_prototype.cpp -o /tmp/mps_prototype
 */
#include "circuit.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using cd = std::complex<double>;

/* Reference: full 65536-amplitude statevector, as the miner does today. */
static void reference_expect(const uint8_t nibbles[QHASH_NIBBLE_COUNT], int offset, double out[16])
{
    std::vector<qhash::Complex<double>> psi(QHASH_STATE_SIZE);
    qhash::init_zero_state(psi.data());
    qhash::run_circuit(psi.data(), nibbles, offset);
    qhash::measure_z_expectations(psi.data(), out);
}

/* Candidate: O(16) sweep over a 2x2 accumulator. */
static void mps_expect(const uint8_t nibbles[QHASH_NIBBLE_COUNT], int offset, double out[16])
{
    cd a[16], b[16];
    double sfac[16], g00[16], g01[16];

    for (int q = 0; q < 16; ++q) {
        const double ty1 = qhash::nibble_angle<double>(nibbles[q], offset);
        const double tz1 = qhash::nibble_angle<double>(nibbles[16 + q], offset);
        const double ty2 = qhash::nibble_angle<double>(nibbles[32 + q], offset);
        /* nibbles[48+q] (layer-2 Rz) is deliberately unused - see claim below. */

        const double c = std::cos(ty1), s = std::sin(ty1);
        const cd ep(std::cos(tz1), std::sin(tz1));
        const cd em(std::cos(tz1), -std::sin(tz1));

        a[q] = ep * c;
        b[q] = -em * s;
        sfac[q] = 2.0 * std::real(a[q] * std::conj(b[q]));
        g00[q] = std::cos(2.0 * ty2);
        g01[q] = std::sin(2.0 * ty2);
    }

    cd L[2][2] = {{cd(1, 0), cd(0, 0)}, {cd(0, 0), cd(0, 0)}};

    for (int q = 0; q < 16; ++q) {
        const cd C[2][2] = {{a[q], b[q]}, {b[q], a[q]}};
        cd t[2][2];
        for (int v = 0; v < 2; ++v) {
            for (int vp = 0; vp < 2; ++vp) {
                cd acc(0, 0);
                for (int u = 0; u < 2; ++u)
                    for (int up = 0; up < 2; ++up)
                        acc += std::conj(C[u][v]) * L[u][up] * C[up][vp];
                t[v][vp] = acc;
            }
        }
        const cd G[2][2] = {{cd(g00[q], 0), cd(g01[q], 0)}, {cd(g01[q], 0), cd(-g00[q], 0)}};
        for (int v = 0; v < 2; ++v)
            for (int vp = 0; vp < 2; ++vp)
                L[v][vp] = t[v][vp] * G[v][vp];

        const double sr = (q < 15) ? sfac[q + 1] : 1.0;
        out[q] = std::real(L[0][0] + L[1][1] + sr * (L[0][1] + L[1][0]));
    }
}

int main(int argc, char** argv)
{
    const int trials = (argc > 1) ? std::atoi(argv[1]) : 200;
    const int bench = (argc > 2) ? std::atoi(argv[2]) : 0;
    std::mt19937 rng(12345);

    if (bench > 0) {
        uint8_t nib[QHASH_NIBBLE_COUNT];
        for (int i = 0; i < QHASH_NIBBLE_COUNT; ++i)
            nib[i] = uint8_t(rng() & 0xF);
        double out[16], sink = 0.0;

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < bench; ++i) {
            nib[i & 63] = uint8_t(i & 0xF);
            mps_expect(nib, 1, out);
            sink += out[0];
        }
        auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();

        auto t2 = std::chrono::steady_clock::now();
        const int sv_iters = 200;
        for (int i = 0; i < sv_iters; ++i) {
            nib[i & 63] = uint8_t(i & 0xF);
            reference_expect(nib, 1, out);
            sink += out[0];
        }
        auto t3 = std::chrono::steady_clock::now();
        const double sv_secs = std::chrono::duration<double>(t3 - t2).count();

        std::printf("closed form : %.0f circuits/s (1 CPU thread)\n", bench / secs);
        std::printf("statevector : %.0f circuits/s (1 CPU thread)\n", sv_iters / sv_secs);
        std::printf("ratio       : %.0fx  (sink %.3f)\n", (bench / secs) / (sv_iters / sv_secs), sink);
        return 0;
    }

    double max_abs = 0.0;
    long q15_total = 0, q15_mismatch = 0;
    long nonce_mismatch = 0;

    for (int t = 0; t < trials; ++t) {
        uint8_t nibbles[QHASH_NIBBLE_COUNT];
        for (int i = 0; i < QHASH_NIBBLE_COUNT; ++i)
            nibbles[i] = uint8_t(rng() & 0xF);
        const int offset = (t & 1);

        double ref[16], got[16];
        reference_expect(nibbles, offset, ref);
        mps_expect(nibbles, offset, got);

        bool bad = false;
        for (int q = 0; q < 16; ++q) {
            max_abs = std::max(max_abs, std::fabs(ref[q] - got[q]));
            ++q15_total;
            if (qhash::to_fixed_q15(ref[q]) != qhash::to_fixed_q15(got[q])) {
                ++q15_mismatch;
                bad = true;
            }
        }
        if (bad)
            ++nonce_mismatch;
    }

    std::printf("trials              : %d\n", trials);
    std::printf("max |dZ|            : %.3e\n", max_abs);
    std::printf("Q1.15 mismatches    : %ld / %ld\n", q15_mismatch, q15_total);
    std::printf("nonce mismatches    : %ld / %d\n", nonce_mismatch, trials);

    /* Secondary claim: layer-2 Rz nibbles (48..63) cannot affect the result. */
    {
        uint8_t nib[QHASH_NIBBLE_COUNT];
        for (int i = 0; i < QHASH_NIBBLE_COUNT; ++i)
            nib[i] = uint8_t(rng() & 0xF);
        double base[16], perturbed[16];
        reference_expect(nib, 1, base);
        for (int i = 48; i < 64; ++i)
            nib[i] = uint8_t((nib[i] + 7) & 0xF);
        reference_expect(nib, 1, perturbed);
        double d = 0.0;
        for (int q = 0; q < 16; ++q)
            d = std::max(d, std::fabs(base[q] - perturbed[q]));
        std::printf("layer-2 Rz influence: %.3e (expect ~0)\n", d);
    }

    return (q15_mismatch == 0) ? 0 : 1;
}
