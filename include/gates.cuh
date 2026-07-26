#pragma once
/**
 * Single-qubit gate primitives + CNOT for 16-qubit statevectors.
 * Matches custatevecApplyPauliRotation / ApplyMatrix semantics used by Qubitcoin.
 */
#include "qhash_params.h"

#include <cstdint>
#include <math.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define QHASH_HD __host__ __device__ __forceinline__
#else
#define QHASH_HD inline
#endif

#ifndef QHASH_PI
#define QHASH_PI 3.14159265358979323846
#endif

namespace qhash {

QHASH_HD double qcos(double x)
{
#ifdef __CUDA_ARCH__
    return cos(x);
#else
    return ::cos(x);
#endif
}
QHASH_HD float qcos(float x)
{
#ifdef __CUDA_ARCH__
    return cosf(x);
#else
    return ::cosf(x);
#endif
}
QHASH_HD double qsin(double x)
{
#ifdef __CUDA_ARCH__
    return sin(x);
#else
    return ::sin(x);
#endif
}
QHASH_HD float qsin(float x)
{
#ifdef __CUDA_ARCH__
    return sinf(x);
#else
    return ::sinf(x);
#endif
}

/**
 * Fused multiply-add, a*b+c with a single rounding.
 *
 * Unlike compiler-contracted a*b+c (which we keep disabled via -ffp-contract=off
 * and --fmad=false because the two compilers fuse different subexpressions), an
 * explicit IEEE-754 fma has exactly one correctly-rounded result. glibc's fma and
 * the device's __fma_rn therefore return identical bits, so the closed-form sweep
 * stays byte-for-byte comparable between CPU and GPU while issuing half as many
 * FP64 instructions.
 */
QHASH_HD double qfma(double a, double b, double c)
{
#ifdef __CUDA_ARCH__
    return __fma_rn(a, b, c);
#else
    return ::fma(a, b, c);
#endif
}

template <typename Real>
struct Complex {
    Real re, im;

    QHASH_HD Complex() : re(0), im(0) {}
    QHASH_HD Complex(Real r, Real i) : re(r), im(i) {}

    QHASH_HD Complex operator+(const Complex& o) const { return Complex(re + o.re, im + o.im); }
    QHASH_HD Complex operator-(const Complex& o) const { return Complex(re - o.re, im - o.im); }
    QHASH_HD Complex operator*(const Complex& o) const
    {
        return Complex(re * o.re - im * o.im, re * o.im + im * o.re);
    }
    QHASH_HD Complex operator*(Real s) const { return Complex(re * s, im * s); }
    QHASH_HD Real norm2() const { return re * re + im * im; }
};

/** 2×2 unitary acting on a qubit amplitude pair. Row-major: [u00 u01; u10 u11]. */
template <typename Real>
struct U2 {
    Complex<Real> u00, u01, u10, u11;
};

/**
 * Build fused Rz(θz) · Ry(θy) matching circuit order (Ry then Rz).
 *
 * IMPORTANT — matches custatevecApplyPauliRotation as implemented by cuStateVec
 * 1.14 (and qubitcoin-node / official miner), empirically:
 *   ApplyPauliRotation(θ, Y) = exp(+i θ Y) = [[c, s], [-s, c]], c=cosθ, s=sinθ
 *   ApplyPauliRotation(θ, Z) = exp(+i θ Z) = diag(e^{+iθ}, e^{-iθ})
 * This is NOT the textbook exp(-i θ/2 P) form (NVIDIA doc string); do not “fix”
 * by inserting θ/2 — consensus digests follow the library behavior above.
 */
template <typename Real>
QHASH_HD U2<Real> make_ry(Real theta_y)
{
    const Real c = qcos(theta_y);
    const Real s = qsin(theta_y);
    U2<Real> u;
    u.u00 = Complex<Real>(c, 0);
    u.u01 = Complex<Real>(s, 0);
    u.u10 = Complex<Real>(-s, 0);
    u.u11 = Complex<Real>(c, 0);
    return u;
}

template <typename Real>
QHASH_HD U2<Real> make_rz(Real theta_z)
{
    const Real cz = qcos(theta_z);
    const Real sz = qsin(theta_z);
    U2<Real> u;
    /* e^{+i θ} = cz + i sz ; e^{-i θ} = cz - i sz */
    u.u00 = Complex<Real>(cz, sz);
    u.u01 = Complex<Real>(0, 0);
    u.u10 = Complex<Real>(0, 0);
    u.u11 = Complex<Real>(cz, -sz);
    return u;
}

template <typename Real>
QHASH_HD U2<Real> make_rz_ry(Real theta_y, Real theta_z)
{
    const Real c = qcos(theta_y);
    const Real s = qsin(theta_y);
    const Real cz = qcos(theta_z);
    const Real sz = qsin(theta_z);
    /* Rz·Ry with exp(+iθP) convention: e^{+iθz} on |0⟩ branch, e^{-iθz} on |1⟩ */
    const Complex<Real> ep(cz, sz);   /* e^{+i θz} */
    const Complex<Real> em(cz, -sz);  /* e^{-i θz} */
    U2<Real> u;
    u.u00 = ep * c;
    u.u01 = ep * s;
    u.u10 = em * Real(-s);
    u.u11 = em * c;
    return u;
}

/**
 * Apply 2×2 unitary on qubit `q` of an N-qubit state (N = QHASH_NUM_QUBITS).
 * Thread-safe when each thread owns a disjoint set of pairs, or serial on host.
 */
template <typename Real>
QHASH_HD void apply_u2(Complex<Real>* psi, int q, const U2<Real>& u)
{
    const int mask = 1 << q;
    const int n = QHASH_STATE_SIZE;
    for (int i = 0; i < n; ++i) {
        if (i & mask)
            continue;
        const int j = i | mask;
        const Complex<Real> a = psi[i];
        const Complex<Real> b = psi[j];
        psi[i] = u.u00 * a + u.u01 * b;
        psi[j] = u.u10 * a + u.u11 * b;
    }
}

/** Parallel-friendly pair update for device kernels (one pair per call). */
template <typename Real>
QHASH_HD void apply_u2_pair(Complex<Real>& a, Complex<Real>& b, const U2<Real>& u)
{
    const Complex<Real> a2 = u.u00 * a + u.u01 * b;
    const Complex<Real> b2 = u.u10 * a + u.u11 * b;
    a = a2;
    b = b2;
}

/**
 * CNOT(control, target) — nearest-neighbour when target = control+1.
 * Pair-index loop: STATE/4 swaps (same tiling as the CUDA kernel).
 */
template <typename Real>
QHASH_HD void apply_cnot(Complex<Real>* psi, int control, int target)
{
    const int npairs = QHASH_STATE_SIZE >> 2;
    const int cmask = 1 << control;
    const int tmask = 1 << target;
    const int q0 = control < target ? control : target;
    const int q1 = control < target ? target : control;
    const int lo_bits = q0;
    const int mid_bits = q1 - q0 - 1;
    const int lo_mask = (1 << lo_bits) - 1;
    const int mid_mask = (1 << mid_bits) - 1;

    for (int p = 0; p < npairs; ++p) {
        const int lo = p & lo_mask;
        const int mid = (p >> lo_bits) & mid_mask;
        const int hi = p >> (lo_bits + mid_bits);
        const int base = lo | (mid << (q0 + 1)) | (hi << (q1 + 1));
        const int i = base | cmask;
        const int j = i | tmask;
        const Complex<Real> tmp = psi[i];
        psi[i] = psi[j];
        psi[j] = tmp;
    }
}

template <typename Real>
QHASH_HD void apply_cnot_pair(Complex<Real>& a, Complex<Real>& b)
{
    const Complex<Real> tmp = a;
    a = b;
    b = tmp;
}

/** ⟨Z⟩ on qubit q = Σ |amp|² · (+1 if bit clear, −1 if set). */
template <typename Real>
QHASH_HD Real expectation_z(const Complex<Real>* psi, int q)
{
    const int mask = 1 << q;
    Real acc = 0;
    for (int i = 0; i < QHASH_STATE_SIZE; ++i) {
        const Real p = psi[i].norm2();
        acc += (i & mask) ? -p : p;
    }
    return acc;
}

} // namespace qhash
