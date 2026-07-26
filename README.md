# qhash-miner

Custom **CUDA FP64** miner for [Qubitcoin](https://github.com/super-quantum/qubitcoin) **qhash** PoW.  
Drops into the [official miner](https://github.com/super-quantum/qubitcoin-miner) via BYOS — same Stratum client, our kernel instead of cuStateVec.

**Default pool:** [Suprnova QTC](https://qtc.suprnova.cc/) — `stratum+tcp://qtc.suprnova.cc:5555`

---

## Why this exists

All 16 ⟨Z⟩ values follow **exactly** from a 16-step sweep over a single 2×2 matrix — **32 bytes** of
state per nonce instead of a 65 536-amplitude statevector, one *thread* per nonce instead of one
*block*, no shared memory and no barriers. This is an algebraic identity, not an approximation: the
trailing CNOT staircase maps `Z_q → Z_0…Z_q`, and the leading staircase on a product state has a
closed-form prefix-XOR (bond-dimension-2) description. See `PLAN.md` → *Phase 6 notes*.

Shipped path: **FP64** closed-form ⟨Z⟩ sweep + SHA-256 midstate, BYOS into the official Stratum
client. Gate convention matches the node (`exp(+i θ P)`, full angle).

### Measured rates (RTX 5060 Laptop, 3072 CUDA cores)

| | Rate |
|--|-----:|
| Sustained kernel (`--benchmark`, multi-second) | **0.287 GH/s** |
| Official-miner BYOS end-to-end | **0.298 GH/s** |
| Live pool, miner-counted | **~0.292 GH/s** |
| Live pool, Suprnova page (same short run) | **0.151 GH/s** |
| Ablation: SHA-256 alone (sweep free) | **1.39 GH/s** ceiling |

The miner/page gap on ~11 shares is unresolved (~2 σ); treat live mining as **0.15–0.29 GH/s** until a
longer run settles it (`PLAN.md` → *Unresolved: pool-side rate*). The kernel is at **96%** of this
card's measured FP64 instruction-issue rate.

### Against the field

Suprnova [publishes per-GPU qhash rates](https://qtc.suprnova.cc/index.php?page=calculator):

| | CUDA cores | Rate | GH/s per 1000 cores |
|--|-----------:|-----:|--------------------:|
| RTX 4070 (pool-listed) | 5888 | 1.97 GH/s | 0.335 |
| RTX 4060 (pool-listed) | 3072 | 0.89 GH/s | 0.290 |
| **This project** (5060 Laptop) | 3072 | **0.15–0.29 GH/s** | 0.05–0.095 |

Same core count as a 4060: we are roughly **3–6× behind**. Reason: FP64-bound (consumer FP64 is
1/64 rate). Ablation: sweep 3.285 ns/nonce, SHA-256 0.717 ns/nonce — a free sweep would hit the
**1.39 GH/s** SHA ceiling and sit above every row above. Listed peer rates fall between our FP64 rate
and that ceiling (reduced-precision sweep + SHA-bound tail). Closing the gap is Phase 8 in `PLAN.md`;
oracle re-verify means a lower-precision *mining* kernel can only cost missed shares, never invalid
ones.

### Correctness

The statevector is the **consensus oracle**, not the mining path:

- Closed form vs FP64 statevector: **0 real divergences** over 4 M nonces / 64 M Q1.15 values.
- Exact Q1.15 boundary ties (~1 in 10⁵) have no unique FP64 answer; every target-passing candidate is
  **re-simulated with the statevector**, and that digest is submitted.
- FP32 mining is **off** for now (Phase 4 statevector match ~92%); Phase 8 reopens reduced precision
  as a *filter* behind the oracle.

Live at Suprnova Diff 0.5 (short run): **11 submitted / 11 accepted / 0 rejected / 0 stale.**

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

Sanity-check the GPU path before pointing it at a pool:

```bash
./scripts/mine-suprnova.sh --benchmark    # no pool, no wallet needed
```

Healthy output ramps over the first ~20 s as the GPU boosts, then settles near **~0.29 GH/s**
(miner prints MH/s: ~290 MH/s) on an RTX 5060 Laptop.

If it settles around **~0.0007 GH/s** (~0.7 MH/s) instead, the CUDA path did not engage and it fell
back to the CPU BYOS; check that `build-byos-cuda/` exists and that `--cuda` was passed to
`build-official-byos.sh`.

Note that `--benchmark` uses an unreachable target, so it never exercises candidate handling. If the
pool rate is far below the benchmark rate, run with `-D` to see the per-batch breakdown:

```
qhash cuda: 4194304 nonces in 0.014 s (~0.29 GH/s kernel), 0 candidates, oracle checks 0 rejected 0
```

`candidates` should be 0 for almost every batch at any real difficulty. A large number means the
target filter is not filtering, and each one costs a ~4.6 ms statevector re-hash.

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

# Sustained hashrate. Needs a multi-second load: a cold 4M launch reads ~0.026 GH/s
# purely because the GPU has not boosted yet.
./build/qhash-miner --benchmark --nonces 4000000000 --chunk 4194304
# RTX 5060 Laptop: ~0.287 GH/s

# Statevector oracle path (A/B / re-verify only — not for mining):
./build/qhash-miner --benchmark --sim statevector --nonces 2048 --threads 256 --chunk 128
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
- **FP32 mining OFF** for now; Phase 8 may reopen reduced precision as a filter behind the oracle

Run the gate with `./build/qhash-closed-form-check --mode all --nonces 4000000`; arbitrate a single
nonce with `--mode nonce --nonce N --offset 0|1`.

---

## Status / limits

- **0.287 GH/s** sustained kernel, **0.298 GH/s** end-to-end BYOS, live **0.15–0.29 GH/s** (pool page
  vs miner count unresolved) on an RTX 5060 **Laptop**.
- **FP64-instruction-bound at 96%** of the card's measured issue rate (379 FP64 instr/nonce). SHA-256
  is ~12% of time and fully hidden; geometry is flat within 1% over threads 128–512 and 1–16 waves.
- Measure with a **multi-second** load (cold 4 M launch ≈ 0.026 GH/s before boost).
- Ablation SHA ceiling **1.39 GH/s**; field peers on same cores ~**0.89 GH/s** — Phase 8 aims to close
  that gap. Long-run accept rate and multi-GPU are in `PLAN.md`.
- Nsight Compute needs Windows **GPU performance counters** enabled under WSL; cost split came from
  `qhash-cf-ablation` instead.

---

## References

- Node: https://github.com/super-quantum/qubitcoin  
- Official miner: https://github.com/super-quantum/qubitcoin-miner  
- Default pool: https://qtc.suprnova.cc/  
