#include "qhash_cpu.h"
#include "circuit.cuh"
#include "closed_form.cuh"
#include "sha256.cuh"

#include <cstring>
#include <vector>

using namespace qhash;

namespace {

template <typename Real>
void statevector_impl(const uint8_t nibbles[QHASH_NIBBLE_COUNT],
                      double expectations[QHASH_NUM_QUBITS],
                      uint32_t nTime)
{
    std::vector<Complex<Real>> psi(QHASH_STATE_SIZE);
    init_zero_state(psi.data());
    run_circuit(psi.data(), nibbles, angle_offset(nTime));
    Real exp_r[QHASH_NUM_QUBITS];
    measure_z_expectations(psi.data(), exp_r);
    for (int i = 0; i < QHASH_NUM_QUBITS; ++i)
        expectations[i] = double(exp_r[i]);
}

void simulate_impl(const uint8_t nibbles[QHASH_NIBBLE_COUNT],
                   double expectations[QHASH_NUM_QUBITS],
                   qhash_precision_t precision,
                   uint32_t nTime,
                   qhash_sim_t sim)
{
    /* The closed form is an FP64 identity; FP32 exists only to reproduce the
       legacy miner's statevector rounding, so it always takes the oracle path. */
    if (precision == QHASH_PRECISION_FP32)
        statevector_impl<float>(nibbles, expectations, nTime);
    else if (sim == QHASH_SIM_STATEVECTOR)
        statevector_impl<double>(nibbles, expectations, nTime);
    else
        closed_form_expectations_host(nibbles, angle_offset(nTime), expectations);
}

uint32_t read_ntime(const uint8_t header[QHASH_INPUT_SIZE])
{
    /* Bitcoin block header: nTime at offset 68, little-endian. */
    return uint32_t(header[68]) | (uint32_t(header[69]) << 8) |
           (uint32_t(header[70]) << 16) | (uint32_t(header[71]) << 24);
}

void hash_impl(const uint8_t header[QHASH_INPUT_SIZE],
               uint8_t out[QHASH_SHA256_SIZE],
               qhash_precision_t precision,
               uint32_t nTime,
               qhash_sim_t sim)
{
    uint8_t in_hash[QHASH_SHA256_SIZE];
    sha256(header, QHASH_INPUT_SIZE, in_hash);

    uint8_t nibbles[QHASH_NIBBLE_COUNT];
    split_nibbles(in_hash, nibbles);

    double expectations[QHASH_NUM_QUBITS];
    simulate_impl(nibbles, expectations, precision, nTime, sim);

    qhash_finalize_cpu(in_hash, expectations, nTime, out);
}

} // namespace

extern "C" void qhash_finalize_cpu(const uint8_t in_hash[QHASH_SHA256_SIZE],
                                   const double expectations[QHASH_NUM_QUBITS],
                                   uint32_t nTime,
                                   uint8_t out[QHASH_SHA256_SIZE])
{
    uint8_t buf[QHASH_SHA256_SIZE + QHASH_NUM_QUBITS * sizeof(int16_t)];
    std::memcpy(buf, in_hash, QHASH_SHA256_SIZE);

    int zeroes = 0;
    for (int i = 0; i < QHASH_NUM_QUBITS; ++i) {
        const int16_t fixed = to_fixed_q15(expectations[i]);
        const size_t j = QHASH_SHA256_SIZE + i * sizeof(int16_t);
        buf[j] = uint8_t(fixed & 0xFF);
        buf[j + 1] = uint8_t((fixed >> 8) & 0xFF);
        if (buf[j] == 0)
            ++zeroes;
        if (buf[j + 1] == 0)
            ++zeroes;
    }

    if (softfork_reject_zeroes(nTime, zeroes)) {
        for (int i = 0; i < QHASH_SHA256_SIZE; ++i)
            out[i] = 0xFF;
        return;
    }

    sha256(buf, sizeof(buf), out);
}

extern "C" void qhash_simulate_cpu_sim(const uint8_t nibbles[QHASH_NIBBLE_COUNT],
                                       double expectations[QHASH_NUM_QUBITS],
                                       qhash_precision_t precision,
                                       uint32_t nTime,
                                       qhash_sim_t sim)
{
    simulate_impl(nibbles, expectations, precision, nTime, sim);
}

/* Default simulator for callers that do not choose one. Promoted to the closed
   form in Phase 6.4 after 6.2/6.3 found zero divergences that any FP64 simulator
   could resolve; the statevector stays reachable via the _sim entry points. */
#ifndef QHASH_DEFAULT_SIM
#define QHASH_DEFAULT_SIM QHASH_SIM_CLOSED_FORM
#endif

extern "C" void qhash_simulate_cpu(const uint8_t nibbles[QHASH_NIBBLE_COUNT],
                                   double expectations[QHASH_NUM_QUBITS],
                                   qhash_precision_t precision,
                                   uint32_t nTime)
{
    simulate_impl(nibbles, expectations, precision, nTime, QHASH_DEFAULT_SIM);
}

extern "C" void qhash_hash_cpu_sim(const uint8_t header[QHASH_INPUT_SIZE],
                                   uint8_t out[QHASH_SHA256_SIZE],
                                   qhash_precision_t precision,
                                   uint32_t force_nTime,
                                   qhash_sim_t sim)
{
    const uint32_t nTime = force_nTime ? force_nTime : read_ntime(header);
    hash_impl(header, out, precision, nTime, sim);
}

extern "C" void qhash_hash_cpu(const uint8_t header[QHASH_INPUT_SIZE],
                               uint8_t out[QHASH_SHA256_SIZE],
                               qhash_precision_t precision,
                               uint32_t force_nTime)
{
    qhash_hash_cpu_sim(header, out, precision, force_nTime, QHASH_DEFAULT_SIM);
}
