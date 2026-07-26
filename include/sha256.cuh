#pragma once
/**
 * Compact SHA-256 for host and device (__device__ / host).
 * Bit-exact vs FIPS 180-4 / Bitcoin SHA256d midstate usage for single-shot hashes.
 */
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define QHASH_HD __host__ __device__ __forceinline__
#else
#define QHASH_HD inline
#endif

#ifdef __CUDACC__
#define QHASH_UNROLL _Pragma("unroll")
#else
#define QHASH_UNROLL
#endif

namespace qhash {

QHASH_HD uint32_t sha_rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
QHASH_HD uint32_t sha_ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
QHASH_HD uint32_t sha_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
QHASH_HD uint32_t sha_bSig0(uint32_t x) { return sha_rotr(x, 2) ^ sha_rotr(x, 13) ^ sha_rotr(x, 22); }
QHASH_HD uint32_t sha_bSig1(uint32_t x) { return sha_rotr(x, 6) ^ sha_rotr(x, 11) ^ sha_rotr(x, 25); }
QHASH_HD uint32_t sha_sSig0(uint32_t x) { return sha_rotr(x, 7) ^ sha_rotr(x, 18) ^ (x >> 3); }
QHASH_HD uint32_t sha_sSig1(uint32_t x) { return sha_rotr(x, 17) ^ sha_rotr(x, 19) ^ (x >> 10); }

QHASH_HD void sha256_transform(uint32_t state[8], const uint8_t block[64])
{
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    uint32_t W[64];
    QHASH_UNROLL
    for (int i = 0; i < 16; ++i) {
        W[i] = (uint32_t(block[4 * i]) << 24) | (uint32_t(block[4 * i + 1]) << 16) |
               (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
    }
    QHASH_UNROLL
    for (int i = 16; i < 64; ++i)
        W[i] = sha_sSig1(W[i - 2]) + W[i - 7] + sha_sSig0(W[i - 15]) + W[i - 16];

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    QHASH_UNROLL
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + sha_bSig1(e) + sha_ch(e, f, g) + K[i] + W[i];
        uint32_t t2 = sha_bSig0(a) + sha_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
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

QHASH_HD void sha256_init(uint32_t state[8])
{
    state[0] = 0x6a09e667u;
    state[1] = 0xbb67ae85u;
    state[2] = 0x3c6ef372u;
    state[3] = 0xa54ff53au;
    state[4] = 0x510e527fu;
    state[5] = 0x9b05688cu;
    state[6] = 0x1f83d9abu;
    state[7] = 0x5be0cd19u;
}

/** One-shot SHA-256. out must be 32 bytes. */
QHASH_HD void sha256(const uint8_t* data, uint64_t len, uint8_t out[32])
{
    uint32_t state[8];
    sha256_init(state);

    uint64_t off = 0;
    while (off + 64 <= len) {
        sha256_transform(state, data + off);
        off += 64;
    }

    uint8_t block[64];
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i)
        block[i] = data[off + i];
    block[rem] = 0x80;
    for (uint64_t i = rem + 1; i < 64; ++i)
        block[i] = 0;

    if (rem >= 56) {
        sha256_transform(state, block);
        for (int i = 0; i < 64; ++i)
            block[i] = 0;
    }

    uint64_t bitlen = len * 8;
    block[63] = uint8_t(bitlen);
    block[62] = uint8_t(bitlen >> 8);
    block[61] = uint8_t(bitlen >> 16);
    block[60] = uint8_t(bitlen >> 24);
    block[59] = uint8_t(bitlen >> 32);
    block[58] = uint8_t(bitlen >> 40);
    block[57] = uint8_t(bitlen >> 48);
    block[56] = uint8_t(bitlen >> 56);
    sha256_transform(state, block);

    QHASH_UNROLL
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = uint8_t(state[i] >> 24);
        out[4 * i + 1] = uint8_t(state[i] >> 16);
        out[4 * i + 2] = uint8_t(state[i] >> 8);
        out[4 * i + 3] = uint8_t(state[i]);
    }
}

} // namespace qhash
