#pragma once
/**
 * SHA-256 specialised for the qhash inner loop (Phase 6.9).
 *
 * With the quantum simulation reduced to a few hundred flops, SHA-256 is the
 * dominant cost, so the three per-nonce compressions are cut down as far as the
 * message layout allows:
 *
 *   - The 80-byte header's first block (bytes 0..63) is nonce-independent, so the
 *     host computes its midstate once per job: 4 compressions become 3.
 *   - The header's second block has only W[3] (the nonce) varying; W[4..15] are
 *     the padding constants and W[0..2] are job constants, so nothing is loaded
 *     or byte-swapped per nonce.
 *   - The 64-byte final hash's second block is padding only, so its entire
 *     W[0..63] schedule is constant. It is computed once on the host and consumed
 *     from constant memory with a uniform index.
 *   - The message schedule uses a rolling 16-word window instead of a 64-word
 *     array, which keeps ~48 registers free for occupancy.
 *
 * All of this is a re-association of the same arithmetic: digests are identical to
 * sha256() in sha256.cuh, which the unit tests assert against the NIST vectors.
 */
#include "sha256.cuh"

namespace qhash {

/** Round constants. Kept function-local so one definition serves host and device. */
QHASH_HD const uint32_t* sha256_k()
{
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};
    return K;
}

/** One SHA-256 round; `kw` is K[i] + W[i]. */
QHASH_HD void sha256_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t& e,
                           uint32_t& f, uint32_t& g, uint32_t& h, uint32_t kw)
{
    const uint32_t t1 = h + sha_bSig1(e) + sha_ch(e, f, g) + kw;
    const uint32_t t2 = sha_bSig0(a) + sha_maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
}

/**
 * Compress one block whose first 16 schedule words are in `w`, expanding the
 * schedule in a rolling 16-word window. `w` is clobbered.
 */
QHASH_HD void sha256_compress(uint32_t state[8], uint32_t w[16])
{
    const uint32_t* K = sha256_k();
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    QHASH_UNROLL
    for (int i = 0; i < 16; ++i)
        sha256_round(a, b, c, d, e, f, g, h, K[i] + w[i]);

    QHASH_UNROLL
    for (int i = 16; i < 64; ++i) {
        /* W[i] = W[i-16] + σ0(W[i-15]) + W[i-7] + σ1(W[i-2]) */
        w[i & 15] += sha_sSig0(w[(i + 1) & 15]) + w[(i + 9) & 15] + sha_sSig1(w[(i + 14) & 15]);
        sha256_round(a, b, c, d, e, f, g, h, K[i] + w[i & 15]);
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/** Compress a block whose full 64-word schedule is already known. */
QHASH_HD void sha256_compress_const(uint32_t state[8], const uint32_t w[64])
{
    const uint32_t* K = sha256_k();
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    QHASH_UNROLL
    for (int i = 0; i < 64; ++i)
        sha256_round(a, b, c, d, e, f, g, h, K[i] + w[i]);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/** Expand W[0..15] into the full W[0..63] schedule (host-side precomputation). */
inline void sha256_expand_schedule(uint32_t w[64])
{
    for (int i = 16; i < 64; ++i)
        w[i] = sha_sSig1(w[i - 2]) + w[i - 7] + sha_sSig0(w[i - 15]) + w[i - 16];
}

/** Big-endian word load, matching SHA-256's message parsing. */
QHASH_HD uint32_t sha256_load_be(const uint8_t* p)
{
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

/**
 * Nonce-independent state after the header's first 64 bytes.
 * The caller must supply at least 64 bytes.
 */
inline void sha256_midstate(const uint8_t header[64], uint32_t midstate[8])
{
    uint32_t w[16];
    for (int i = 0; i < 16; ++i)
        w[i] = sha256_load_be(header + 4 * i);
    sha256_init(midstate);
    sha256_compress(midstate, w);
}

/**
 * Schedule for the padding-only second block of a message of `len` bytes whose
 * data occupies a whole number of blocks (here: the 64-byte final hash).
 */
inline void sha256_pad_schedule(uint32_t w[64], uint64_t len)
{
    for (int i = 0; i < 16; ++i)
        w[i] = 0;
    w[0] = 0x80000000u;
    w[14] = uint32_t((len * 8) >> 32);
    w[15] = uint32_t(len * 8);
    sha256_expand_schedule(w);
}

} // namespace qhash
