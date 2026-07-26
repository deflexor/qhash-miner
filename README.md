# qhash-miner

Custom **CUDA FP64** miner for [Qubitcoin](https://github.com/super-quantum/qubitcoin) **qhash** PoW.  
Drops into the [official miner](https://github.com/super-quantum/qubitcoin-miner) via BYOS — same Stratum client, our kernel instead of cuStateVec.

**Default pool:** [Suprnova QTC](https://qtc.suprnova.cc/) — `stratum+tcp://qtc.suprnova.cc:5555`

---

## Why this exists

The stock miner launches **cuStateVec once per gate** (FP32). We replace that with a **monolithic FP64 CUDA kernel** (fused gates, shared-memory tiling, fused ⟨Z⟩ + SHA-256) that matches consensus bit-exactness.

| | Official (cuStateVec) | This project (BYOS CUDA) |
|--|----------------------|---------------------------|
| Precision | FP32 | **FP64** (consensus) |
| Same-card rate (RTX 5060 Laptop) | **~0.76 kh/s** | **~3.1–3.3 kh/s** |
| Speedup | 1× | **~4×** |
| Gate convention | cuStateVec `exp(+i θ P)` | same (verified) |

Literature baselines (other GPUs): official ~4.5 kh/s on RTX 4070; community AllFather ~20 kh/s. Our current plateau on a 5060 Laptop is **~3 kh/s** — still a large win vs stock on the *same* card; further gains need Nsight Compute (GPU counters) or larger algorithmic changes.

### Profitability (rough)

Pool payout scales with **accepted shares ≈ hashrate × time** (same pool/diff).

- **Revenue:** ~**4×** vs stock cuStateVec on the same GPU (measured hashrate ratio).
- **Power:** FP64 usually draws more than FP32; net **profit** uplift is typically a bit under the hashrate ratio (measure with `nvidia-smi` on your rig).
- **Correctness:** FP32 mining is **disabled** here — digest match vs FP64 was only ~92% over 1.5M nonces (need ≥99.99% for safe mining).

At Suprnova Diff **0.5** and ~3 kh/s, expect **~8–9 days** between shares. Hash validation was already proven (11/11 submits → only `low difficulty share` rejects via a low-diff proxy; 0 invalid).

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
├── include/           # circuit, gates, sha256, params
├── src/
│   ├── qhash_kernel.cu      # monolithic CUDA miner
│   ├── qhash_cpu.cpp        # FP64 reference
│   └── byos/qhash_byos.cpp  # official-miner drop-in
├── scripts/
│   ├── build-official-byos.sh
│   ├── mine-suprnova.sh     # default pool helper
│   └── pool-accept-check.sh
├── tests/
├── PLAN.md            # detailed progress / benchmarks
└── AGENTS.md          # toolchain notes for agents
```

---

## Correctness

- Self-test + `ctest` / `qhash-test`  
- vs cuStateVec FP64 Q1.15: **1024/1024** bit-exact after gate-convention fix  
- GPU vs CPU FP64 digests: bit-exact on spot checks  
- **FP32 mining OFF** (~92% digest match — unsafe)

---

## Status / limits

- Kernel plateau ~**2.7–3.3 kh/s** on RTX 5060 Laptop (tile 2048, T=256, C=128).  
- Nsight Compute needs Windows **GPU performance counters** enabled (WSL: NVIDIA App / Control Panel → Developer → allow counters).  
- Multi-GPU and further kernel work tracked in `PLAN.md`.

---

## References

- Node: https://github.com/super-quantum/qubitcoin  
- Official miner: https://github.com/super-quantum/qubitcoin-miner  
- Default pool: https://qtc.suprnova.cc/  
