/**
 * Micro-benchmark focused on circuit simulation throughput (CPU path always;
 * CUDA when linked).
 */
#include "qhash_cpu.h"
#include "qhash_kernel.cuh"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv)
{
    uint32_t n = 64;
    if (argc > 1)
        n = uint32_t(std::strtoul(argv[1], nullptr, 10));

    uint8_t header[QHASH_INPUT_SIZE];
    std::memset(header, 0x11, sizeof(header));
    const uint32_t nTime = QHASH_SF_ANGLE;
    header[68] = uint8_t(nTime);
    header[69] = uint8_t(nTime >> 8);
    header[70] = uint8_t(nTime >> 16);
    header[71] = uint8_t(nTime >> 24);

    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < n; ++i) {
        header[76] = uint8_t(i);
        header[77] = uint8_t(i >> 8);
        header[78] = uint8_t(i >> 16);
        header[79] = uint8_t(i >> 24);
        uint8_t out[32];
        qhash_hash_cpu(header, out, QHASH_PRECISION_FP64, nTime);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    std::printf("CPU FP64: %u hashes in %.3fs → %.2f H/s\n", n, s, n / s);

    t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < n; ++i) {
        header[76] = uint8_t(i);
        uint8_t out[32];
        qhash_hash_cpu(header, out, QHASH_PRECISION_FP32, nTime);
    }
    t1 = std::chrono::steady_clock::now();
    const double s2 = std::chrono::duration<double>(t1 - t0).count();
    std::printf("CPU FP32: %u hashes in %.3fs → %.2f H/s\n", n, s2, n / s2);
    return 0;
}
