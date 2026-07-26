#pragma once
/**
 * Long-double arbiter for Q1.15 disagreements between FP64 simulators.
 *
 * Written straight from the Phase-6 derivation rather than from
 * closed_form.cuh, so it also cross-checks that header's algebra. Its purpose is
 * to answer one question: when two FP64 simulators put a ⟨Z⟩ value on opposite
 * sides of a Q1.15 rounding boundary, is the *exact* value on one side (a real
 * divergence) or is it the boundary itself (an intrinsically ambiguous input, on
 * which the consensus node is also just rounding noise)?
 *
 * long double carries a 64-bit mantissa on x86-64, so its residual is ~40 ulp
 * ≈ 1e-22 absolute here — five orders of magnitude tighter than the FP64
 * statevector's ~1e-12, which is enough to resolve the question.
 */
#include "qhash_params.h"

#include <cmath>
#include <cstdint>

namespace qhash_test {

struct LutLd {
    long double c1[16], s1[16], c2[16], s2[16];
};

/** Cached long-double trig tables; 64 cosl/sinl calls per case would dominate. */
inline const LutLd& lut_ld(int offset)
{
    struct Cache {
        LutLd l[2];
        Cache()
        {
            const long double pi = 3.141592653589793238462643383279502884L;
            for (int o = 0; o < 2; ++o)
                for (int n = 0; n < 16; ++n) {
                    const long double t = -(long double)(2 * n + o) * pi / 32.0L;
                    l[o].c1[n] = cosl(t);
                    l[o].s1[n] = sinl(t);
                    l[o].c2[n] = cosl(2.0L * t);
                    l[o].s2[n] = sinl(2.0L * t);
                }
        }
    };
    static const Cache cache;
    return cache.l[offset ? 1 : 0];
}

/** All 16 ⟨Z⟩ in long double. Reads only nibbles[0..47]; layer-2 Rz is dead. */
inline void reference_sweep_ld(const uint8_t nibbles[QHASH_NIBBLE_COUNT], int offset,
                               long double out[QHASH_NUM_QUBITS])
{
    constexpr int kQ = QHASH_NUM_QUBITS;
    const LutLd& lut = lut_ld(offset);
    const long double* c1 = lut.c1;
    const long double* s1 = lut.s1;
    const long double* c2 = lut.c2;
    const long double* s2 = lut.s2;

    /* Λ as a full 2x2 complex matrix — no Hermitian shortcut, so a bug in the
       production version's Hermitian reduction would show up as a mismatch. */
    long double lr[2][2] = {{1, 0}, {0, 0}};
    long double li[2][2] = {{0, 0}, {0, 0}};

    for (int q = 0; q < kQ; ++q) {
        const int ny = nibbles[q] & 0xF;
        const int nz = nibbles[kQ + q] & 0xF;
        /* a = e^{+iθz}·cos θy, b = −e^{−iθz}·sin θy; C = [[a,b],[b,a]] */
        const long double cz = c1[nz], sz = s1[nz];
        const long double ar = c1[ny] * cz, ai = c1[ny] * sz;
        const long double br = -s1[ny] * cz, bi = s1[ny] * sz;
        const long double cr[2][2] = {{ar, br}, {br, ar}};
        const long double ci[2][2] = {{ai, bi}, {bi, ai}};

        long double tr[2][2] = {{0, 0}, {0, 0}};
        long double ti[2][2] = {{0, 0}, {0, 0}};
        for (int v = 0; v < 2; ++v)
            for (int vp = 0; vp < 2; ++vp)
                for (int u = 0; u < 2; ++u)
                    for (int up = 0; up < 2; ++up) {
                        /* conj(C[u][v]) · Λ[u][up] · C[up][vp] */
                        const long double xr = cr[u][v], xi = -ci[u][v];
                        const long double yr = lr[u][up], yi = li[u][up];
                        const long double zr = cr[up][vp], zi = ci[up][vp];
                        const long double mr = xr * yr - xi * yi;
                        const long double mi = xr * yi + xi * yr;
                        tr[v][vp] += mr * zr - mi * zi;
                        ti[v][vp] += mr * zi + mi * zr;
                    }

        const int ny2 = nibbles[2 * kQ + q] & 0xF;
        const long double g[2][2] = {{c2[ny2], s2[ny2]}, {s2[ny2], -c2[ny2]}};
        for (int v = 0; v < 2; ++v)
            for (int vp = 0; vp < 2; ++vp) {
                lr[v][vp] = tr[v][vp] * g[v][vp];
                li[v][vp] = ti[v][vp] * g[v][vp];
            }

        long double sr = 1.0L;
        if (q + 1 < kQ) {
            const int nyn = nibbles[q + 1] & 0xF;
            const int nzn = nibbles[kQ + q + 1] & 0xF;
            sr = -s2[nyn] * c2[nzn];
        }
        out[q] = lr[0][0] + lr[1][1] + sr * (lr[0][1] + lr[1][0]);
    }
}

/**
 * Distance from x to the nearest value that would round to a different int16,
 * measured in Q1.15 units. to_fixed_q15 truncates x·32768 ± 0.5, so the decision
 * boundary is where that expression crosses an integer.
 */
inline long double q15_boundary_margin_ld(long double x)
{
    const long double mult = 32768.0L;
    const long double y = (x >= 0) ? (x * mult + 0.5L) : (x * mult - 0.5L);
    const long double f = y - floorl(y);
    return f < (1.0L - f) ? f : (1.0L - f);
}

/**
 * A disagreement is "a tie" when the exact value is closer to the boundary than
 * any FP64 simulator could resolve. FP64 ⟨Z⟩ carries ~1e-14 absolute error, which
 * is ~3e-10 Q1.15 units; 1e-15 is far below that and far above the long-double
 * residual, so the classification is unambiguous in both directions.
 */
constexpr long double kQ15TieThreshold = 1e-15L;

inline bool q15_is_tie(const uint8_t nibbles[QHASH_NIBBLE_COUNT], int offset, int qubit,
                       long double* exact_out, long double* margin_out)
{
    long double exact[QHASH_NUM_QUBITS];
    reference_sweep_ld(nibbles, offset, exact);
    const long double margin = q15_boundary_margin_ld(exact[qubit]);
    if (exact_out)
        *exact_out = exact[qubit];
    if (margin_out)
        *margin_out = margin;
    return margin < kQ15TieThreshold;
}

} // namespace qhash_test
