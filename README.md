# qhash-miner

Custom **CUDA FP64** miner for [Qubitcoin](https://github.com/super-quantum/qubitcoin) **qhash** PoW.  
Drops into the [official miner](https://github.com/super-quantum/qubitcoin-miner) via BYOS — same Stratum client, our kernel instead of cuStateVec.

**Default pool:** [Suprnova QTC](https://qtc.suprnova.cc/) — `stratum+tcp://qtc.suprnova.cc:5555`

---

## Why this exists

The stock miner simulates the 16-qubit circuit by pushing **65536 amplitudes** through cuStateVec, one
launch per gate, in FP32. We don't simulate the circuit at all.

All 16 ⟨Z⟩ values follow **exactly** from a 16-step sweep over a single 2×2 matrix — **32 bytes** of
state per nonce instead of 1 MiB, one *thread* per nonce instead of one *block*, no shared memory and
no barriers. This is an algebraic identity, not an approximation: the trailing CNOT staircase maps
`Z_q → Z_0…Z_q`, and the leading staircase on a product state has a closed-form prefix-XOR
(bond-dimension-2) description. See `PLAN.md` → *Phase 6 notes* for the derivation.

| | Official (cuStateVec) | This project (BYOS CUDA) |
|--|----------------------|---------------------------|
| Precision | FP32 | **FP64** (consensus) |
| Method | 65536-amplitude statevector | **closed-form ⟨Z⟩ sweep** (4 doubles) |
| Same-card rate (RTX 5060 **Laptop**) | **~0.76 kh/s** | **~298 000 kh/s** (~298 Mh/s) |
| Speedup | 1× | **~390×** |
| Gate convention | cuStateVec `exp(+i θ P)` | same (verified) |

That is ~**88 000×** our own previous FP64 statevector kernel (~3.25 kh/s), and it puts the kernel at
**96% of this card's measured FP64 instruction-issue rate** — the arithmetic is now the wall, so
there is little left to win without changing precision, which correctness forbids (below).

Literature baselines (other GPUs): official ~4.5 kh/s on RTX 4070; community AllFather ~20 kh/s.

### Correctness

The statevector did not go away — it is the **consensus oracle**:

- Closed form vs FP64 statevector: **0 real divergences** over 4 M nonces / 64 M Q1.15 values, and
  it is **7.5× closer to cuStateVec** than our own statevector was.
- A handful of nonces (~1 in 10⁵) land on an **exact** Q1.15 rounding boundary where no FP64
  evaluation order has a defined answer, including the node's. Every candidate that passes target is
  therefore **re-simulated with the statevector**, and the oracle's digest is the one submitted — so
  the closed form can never decide a share.
- FP32 mining stays **disabled**: digest match vs FP64 was only ~92% over 1.5 M nonces. Float-float
  FP32 was also measured and **rejected** — the Q1.15 safety margin is only ~20× the FP64 residual,
  and float-float is ~128× less accurate.

### Profitability (rough)

Pool payout scales with **accepted shares ≈ hashrate × time** (same pool/diff).

- **Revenue:** ~**390×** vs stock cuStateVec on the same GPU (measured hashrate ratio).
- **Power:** unchanged board power for ~390× the work, so efficiency scales with the ratio.
- At Suprnova Diff **0.5**, the old ~8–9 day wait between shares becomes **sub-second**. Hash
  validation was already proven at 3 kh/s (11/11 submits → only `low difficulty share` rejects via a
  low-diff proxy; 0 invalid); credited accepts at this rate are Phase 7.

---

## Requirements

- NVIDIA GPU + recent driver  
- **CUDA ≥ 12.8** for RTX 50-series (`sm_120`); this repo is developed with **CUDA 13.3**  
- `cmake` ≥ 3.18, `g++`, and (for pool mining) a build of `official-miner` with our BYOS  
- Optional: cuStateVec (cuQuantum) for the golden harness only  

---

## Quick start — pool mining (Suprnova)

```bash
# 1) Toolchain (example paths — adjust to your CUDA install)
export CUDA_HOME=$HOME/opt/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 2) Build this repo + wire into official miner
cmake -S . -B build-byos-cuda -DQHASH_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build build-byos-cuda -j
./scripts/build-official-byos.sh --cuda

# 3) Mine (default pool = Suprnova). Use -t 1 with CUDA.
export WALLET=bc1qYourQubitcoinAddressHere
./scripts/mine-suprnova.sh
# or:
# LD_LIBRARY_PATH=$PWD/build-byos-cuda:$CUDA_HOME/lib \
#   ../official-miner/qubitcoin-miner -a qhash \
#   -o stratum+tcp://qtc.suprnova.cc:5555 \
#   -u "$WALLET.qhashbyos" -p x -t 1
```

Stats: paste your address on https://qtc.suprnova.cc/

Other pools:

```bash
./scripts/pool-accept-check.sh --pool suprnova      # default
./scripts/pool-accept-check.sh --pool herominers --diff 32
./scripts/pool-accept-check.sh --pool luckypool
```

---

## Build (standalone kernel / tests)

```bash
export CUDA_HOME=$HOME/opt/cuda-13.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

cmake -S . -B build -DQHASH_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build build -j

./build/qhash-test
./build/qhash-miner --self-test
# expect digest: 8711a7a489f9021a8d8011434e374944ee4051851d21c9fc08dee11757d25bc4
./build/qhash-miner --benchmark --nonces 2048 --threads 256 --chunk 128
```

CPU-only fallback (no GPU / no toolkit): builds without CUDA; much slower.

---

## Algorithm (consensus)

| Item | Value |
|------|--------|
| Classical hash | **SHA-256** |
| Circuit | 16 qubits × **2 layers**: Ry, Rz per qubit, then NN CNOT |
| Angles | `θ = −(2·nibble + offset)·π/32` (`offset=1` if `nTime ≥ 1758762000`) |
| Gates | cuStateVec-style **`exp(+i θ P)`** (full angle) — not textbook `θ/2` |
| Measure | Pauli **⟨Z⟩** → Q1.15 `int16` LE × 16 |
| Final | `SHA256(header_hash ‖ fixed_expectations)` |
| Mining precision | **FP64 only** in this tree |

---

## Layout

```
qhash-miner/
├── include/
│   ├── closed_form.cuh      # the shipped ⟨Z⟩ sweep (host + device)
│   ├── sha256_mine.cuh      # midstate + constant-folded schedules
│   └── circuit, gates, sha256, params
├── src/
│   ├── qhash_kernel.cu      # closed-form miner + statevector oracle
│   ├── qhash_cpu.cpp        # CPU reference, either simulator
│   └── byos/                # official-miner drop-in (ABI) + CUDA scanhash
├── bench/cf_ablation.cu     # cost split + FP64 roofline probe
├── scripts/
│   ├── build-official-byos.sh
│   ├── mine-suprnova.sh     # default pool helper
│   └── pool-accept-check.sh
├── tests/                   # closed_form_check (the Phase 6 gate), golden, ABI
├── PLAN.md            # detailed progress / benchmarks
└── AGENTS.md          # toolchain notes for agents
```

---

## Correctness

- Self-test + `ctest` (6/6: ABI, circuit, self-test, closed form, soft-forks, cuStateVec golden)
- vs cuStateVec FP64 Q1.15: **1024/1024** bit-exact after the gate-convention fix; the closed form
  re-validated on 20 000 random + 4096 uniform nibble patterns
- Closed form vs FP64 statevector: **0 real divergences** / 64 M Q1.15 values over 4 M nonces
- Closed-form CUDA kernel vs CPU: **bit-exact on 16 M nonces**
- Adversarial coverage: all 8192 uniform nibble patterns, including **1152** exact ±1 values and
  **732** deliberate `+32768 → −32768` int16 wraps, plus an importance-sampled rounding-boundary scan
- Every share re-verified through the statevector oracle before submission
- **FP32 mining OFF** (~92% digest match — unsafe); float-float FP32 measured and rejected

Run the gate with `./build/qhash-closed-form-check --mode all --nonces 4000000`; arbitrate a single
nonce with `--mode nonce --nonce N --offset 0|1`.

---

## Status / limits

- **~287 Mh/s sustained**, ~298 Mh/s end-to-end through official-miner, on an RTX 5060 **Laptop**.
- The kernel is **FP64-instruction-bound at 96%** of the card's measured issue rate (379 FP64
  instructions per nonce). Occupancy, SHA-256 (12% of time, fully hidden) and memory are all
  irrelevant now — geometry is flat within 1% over threads 128–512 and 1–16 resident waves.
- Measure with a **multi-second** load: a cold single launch reads ~26 Mh/s purely because the GPU has
  not boosted yet.
- Nsight Compute still needs Windows **GPU performance counters** enabled (WSL: NVIDIA App / Control
  Panel → Developer → allow counters); the cost split above was obtained by ablation
  (`qhash-cf-ablation`) instead.
- Credited pool shares at this rate, long-run stability, and multi-GPU are tracked in `PLAN.md`.

---

## References

- Node: https://github.com/super-quantum/qubitcoin  
- Official miner: https://github.com/super-quantum/qubitcoin-miner  
- Default pool: https://qtc.suprnova.cc/  
