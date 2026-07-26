#pragma once
/**
 * Closed-form ⟨Z⟩ for the qhash circuit (Phase 6).
 *
 * This is an exact identity, not a truncation: the 65536-amplitude statevector
 * is unnecessary. See "Phase 6 notes" in PLAN.md for the derivation. Summary:
 *
 *   1. The trailing CNOT staircase maps Z_q → Z_0 Z_1 … Z_q (Heisenberg), so the
 *      digest only needs parity expectations before that staircase.
 *   2. Those observables are all-Z and Rz is diagonal, so the layer-2 Rz gates
 *      cancel — nibbles[48..63] are dead inputs.
 *   3. The layer-1 staircase on a product state is a bond-dimension-2 MPS with an
 *      explicit prefix-XOR form, and the all-identity right environment collapses
 *      to (1, s, s, 1). No backward sweep, no environment storage.
 *
 * What is left is a 16-step sweep of one 2×2 Hermitian accumulator Λ:
 *
 *   Λ ← [[1,0],[0,0]]
 *   for q = 0…15:  Λ ← G_q ⊙ (C_q^H Λ C_q);  ⟨Z_q⟩ = Λ00 + Λ11 + s_{q+1}(Λ01+Λ10)
 *
 * Λ Hermitian ⇒ 4 doubles of state for the whole simulation.
 *
 * Determinism: every angle enters through AngleLut, which is always built with
 * host libm and uploaded verbatim to the device. Combined with contraction
 * disabled (`-ffp-contract=off` / `--fmad=false`, set in CMakeLists) the host and
 * device sweeps are bit-identical, so GPU digests can be compared byte-for-byte
 * against the CPU reference.
 */
#include "circuit.cuh"

namespace qhash {

/**
 * Everything the sweep needs about a nibble angle θ = −(2·nibble + offset)·π/32.
 *
 * Only 16 angles exist, so every per-qubit quantity that depends on a single
 * nibble is tabulated rather than recomputed: cos²θ, sin²θ and cosθ·sinθ are
 * stored as the very same products the sweep used to evaluate inline, so moving
 * them into the table is bit-for-bit neutral and simply removes three FP64
 * multiplies per qubit from an FP64-issue-bound kernel.
 */
struct AngleLut {
    double c2[16]; /* cos 2θ, also cos²θ − sin²θ */
    double s2[16]; /* sin 2θ */
    double na[16]; /* cos²θ  */
    double nb[16]; /* sin²θ  */
    double cs[16]; /* cos θ · sin θ */
};

enum { QHASH_ANGLE_LUT_DOUBLES = sizeof(AngleLut) / sizeof(double) };

/** Fill an AngleLut for one angle offset (0 = legacy miner, 1 = post-soft-fork). */
inline void build_angle_lut(AngleLut& lut, int offset)
{
    for (int n = 0; n < 16; ++n) {
        const double t = nibble_angle<double>(uint8_t(n), offset);
        const double c = ::cos(t);
        const double s = ::sin(t);
        lut.c2[n] = ::cos(2.0 * t);
        lut.s2[n] = ::sin(2.0 * t);
        lut.na[n] = c * c;
        lut.nb[n] = s * s;
        lut.cs[n] = c * s;
    }
}

/** Cached host LUTs, one per angle offset. */
inline const AngleLut& host_angle_lut(int offset)
{
    struct Cache {
        AngleLut l[2];
        Cache()
        {
            build_angle_lut(l[0], 0);
            build_angle_lut(l[1], 1);
        }
    };
    static const Cache cache;
    return cache.l[offset ? 1 : 0];
}

/**
 * Layer-1 factors for one qubit, derived from its Ry/Rz nibbles.
 *
 * With (a, b) = U2_q|0⟩ = (e^{+iθz}·cos θy, −e^{−iθz}·sin θy):
 *   na = |a|², nb = |b|², w = conj(a)·b = −cos θy·sin θy·(cos 2θz − i·sin 2θz)
 * and the environment factor is s_q = 2·Re(a·conj(b)) = 2·w.re.
 *
 * The sweep also needs na ± nb. Those are cos²θ ± sin²θ, i.e. exactly 1 and
 * exactly cos 2θ, so they are taken as the literal 1.0 and straight from the
 * double-angle table instead of being recomputed from the products: fewer FP64
 * instructions, and closer to the exact value than c·c ± s·s.
 */
struct QubitFactors {
    double na, nb, ndif, wr, wi;
};

QHASH_HD QubitFactors qubit_factors(const AngleLut& lut, int ny, int nz)
{
    const double cs = lut.cs[ny];
    QubitFactors f;
    f.na = lut.na[ny];
    f.nb = lut.nb[ny];
    f.ndif = lut.c2[ny]; /* = cos²θy − sin²θy */
    f.wr = -cs * lut.c2[nz];
    f.wi = cs * lut.s2[nz];
    return f;
}

/**
 * Nibble accessors. The sweep only ever reads nibbles 0..47, so the GPU can serve
 * them straight out of the six relevant SHA-256 state words and never materialise
 * a 64-byte per-thread buffer.
 */
struct NibbleBytes {
    const uint8_t* p;
    QHASH_HD int operator()(int i) const { return p[i] & 0xF; }
};

/** Big-endian SHA-256 words, 8 nibbles per word, most significant first. */
struct NibbleWords {
    const uint32_t* w;
    QHASH_HD int operator()(int i) const
    {
        return int((w[i >> 3] >> (28 - 4 * (i & 7))) & 0xFu);
    }
};

/**
 * All 16 ⟨Z⟩ from the nibbles supplied by `nib`. Reads nibbles 0..47 only —
 * nibbles[48..63] are the layer-2 Rz angles, which provably cannot reach the
 * digest.
 *
 * The sweep is software-pipelined: each iteration computes the *next* qubit's
 * factors, which serve both as that qubit's transfer matrix and as this qubit's
 * right-environment factor s_{q+1}.
 */
template <typename NibbleSource>
QHASH_HD void closed_form_sweep(const AngleLut& lut, NibbleSource nib,
                                double out[QHASH_NUM_QUBITS])
{
    constexpr int kQ = QHASH_NUM_QUBITS;

    double l00 = 1.0, l11 = 0.0, l01r = 0.0, l01i = 0.0;
    QubitFactors cur = qubit_factors(lut, nib(0), nib(kQ));

    for (int q = 0; q < kQ; ++q) {
        /* T = C_q^H Λ C_q, again Hermitian. Written as explicit fma chains: the
           kernel is FP64-issue-bound (Phase 6.11 profile: 95% of this card's
           non-fused FP64 rate), and fusing halves the instruction count without
           costing determinism. */
        const double p = cur.wr * l01r;
        const double wd = qfma(-cur.wi, l01i, p); /* wr·Λ01r − wi·Λ01i */
        const double ws = qfma(cur.wi, l01i, p);  /* wr·Λ01r + wi·Λ01i */
        const double t00 = qfma(l00, cur.na, qfma(l11, cur.nb, wd + wd));
        const double t11 = qfma(l00, cur.nb, qfma(l11, cur.na, ws + ws));
        const double t01r = qfma(cur.wr, l00 + l11, l01r); /* na + nb = 1 */
        const double t01i = qfma(cur.wi, l00 - l11, l01i * cur.ndif);

        /* Λ ← G_q ⊙ T with G_q = [[cos 2θy2, sin 2θy2], [sin 2θy2, −cos 2θy2]]. */
        const int ny2 = nib(2 * kQ + q);
        const double g00 = lut.c2[ny2];
        const double g01 = lut.s2[ny2];
        l00 = g00 * t00;
        l11 = -g00 * t11;
        l01r = g01 * t01r;
        l01i = g01 * t01i;

        /* ⟨Z_q⟩ = Λ00 + Λ11 + s_{q+1}·(Λ01 + Λ10), with s_{q+1} = 2·wr of the next
           qubit and Λ01 + Λ10 = 2·Λ01r; the trailing qubit sees s = 1. */
        const double diag = l00 + l11;
        if (q + 1 < kQ) {
            cur = qubit_factors(lut, nib(q + 1), nib(kQ + q + 1));
            out[q] = qfma(cur.wr, 4.0 * l01r, diag);
        } else {
            out[q] = diag + (l01r + l01r);
        }
    }
}

QHASH_HD void closed_form_expectations(const AngleLut& lut,
                                       const uint8_t nibbles[QHASH_NIBBLE_COUNT],
                                       double out[QHASH_NUM_QUBITS])
{
    closed_form_sweep(lut, NibbleBytes{nibbles}, out);
}

/** Convenience wrapper for host callers that only have an angle offset. */
inline void closed_form_expectations_host(const uint8_t nibbles[QHASH_NIBBLE_COUNT], int offset,
                                          double out[QHASH_NUM_QUBITS])
{
    closed_form_expectations(host_angle_lut(offset), nibbles, out);
}

} // namespace qhash
