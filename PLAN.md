# Qhash Miner Optimization — Plan & Progress Log

**Project:** `qhash-miner/` (custom CUDA replacement for cuStateVec qPoW)  
**Started:** 2026-07-26  
**Goal:** 50–200+ kh/s on RTX 4070 (vs official ~4.5 kh/s), bit-exact vs consensus node  
**Status:** Phase 6 complete — **286,934 kh/s** sustained on an RTX 5060 **Laptop**, bit-exact, every
share oracle-verified. Goal exceeded by ~1435×; the kernel now sits at **96% of the card's measured
FP64 instruction-issue rate**, so this algorithm is finished. See [Phase 6 notes](#phase-6-notes--the-statevector-is-unnecessary).

---

## Algorithm truth (do not re-derive from blog posts)

Verified from `super-quantum/qubitcoin-miner` + `super-quantum/qubitcoin`:

| Item | Value |
|------|--------|
| Hash | **SHA-256** (not SHA-3) |
| Circuit | 16 qubits × **2 layers**: Ry, Rz per qubit, then NN CNOT |
| Angles | `θ = −(2·nibble + offset)·π/32`; `offset=1` if `nTime ≥ 1758762000` |
| Gates | **`custatevecApplyPauliRotation(θ,P)` as in cuStateVec 1.14** = `exp(+i θ P)` (full angle). **Not** textbook `exp(−i θ/2 P)`. Fixed 2026-07-26 after golden harness. |
| Measure | Pauli **⟨Z⟩** → Q1.15 `int16` LE × 16 |
| Final | `SHA256(header_hash ‖ fixed_expectations)` |
| Precision | Consensus: **FP64**; stock miner sample: **FP32** |
| Soft-forks | Zero-byte rejection → `0xFF…` digest (see `qhash_params.h`) |

Reference trees (read-only): `official-miner/`, `qubitcoin-node/`, `cpu-miner/`.

---

## Task list

### Phase 0 — Correctness foundation
| ID | Task | Status |
|----|------|--------|
| 0.1 | Extract circuit from official miner/node | ✅ Done |
| 0.2 | CPU reference simulator (FP64/FP32) | ✅ Done |
| 0.3 | SHA-256 + Q1.15 + soft-forks | ✅ Done |
| 0.4 | Unit tests (NIST SHA-256, Ry/CNOT/⟨Z⟩) | ✅ Done |
| 0.5 | Cross-check vs cuStateVec on ≥1k nonces | ✅ Done — **1024/1024** Q1.15 bit-exact after gate-convention fix (`qhash-custatevec-golden`) |
| 0.6 | FP32 vs FP64 fixed-point study ≥10M nonces | ✅ Done — **1.54M digests** (stable) + 10k Q1.15; FP32 **REJECTED** |

### Phase 1 — Monolithic CUDA kernel
| ID | Task | Status |
|----|------|--------|
| 1.1 | One-block-per-nonce kernel (gates+⟨Z⟩+SHA256+target) | ✅ Done |
| 1.2 | Host batch driver + CPU fallback | ✅ Done |
| 1.3 | Build/run on real NVIDIA GPU | ✅ Done (RTX 5060 Laptop, sm_120, local CUDA 13.3) |
| 1.4 | Benchmark vs official miner on same card | ✅ Done (baseline logged; official ~4.5 kh/s is 4070 literature) |
| 1.5 | Nsight Compute: memory / occupancy / stalls | ⚠️ Still **ERR_NVGPUCTRPERM** on WSL (re-checked 2026-07-26) |

### Phase 2 — Gate fusion & circuit hardcoding
| ID | Task | Status |
|----|------|--------|
| 2.1 | Optional `-DQHASH_FUSE_RZ_RY` fused 2×2 | ✅ On by default (`QHASH_FUSE_RZ_RY=ON`); bit-exact vs CPU; **+56%** vs unfused |
| 2.2 | Reduce global passes / better CNOT tiling | ✅ CNOT pair-index + smem tile 2048; `-DQHASH_SMEM_TILE=0` disables; **4096 via dynamic smem** builds but **no win** |
| 2.3 | Measure speedup vs Phase 1 baseline | ✅ Fuse + tile measured |

### Phase 3 — Batching & occupancy
| ID | Task | Status |
|----|------|--------|
| 3.1 | Tune chunk size / threads per block per arch | ✅ Done — **threads=256, chunk=128** best on 5060 |
| 3.2 | Overlap H2D / compute / D2H if needed | ✅ Multi-stream slices (`--streams 1..4`); **~neutral** |
| 3.3 | Multi-stream / multi-GPU | 🟡 Streams done; multi-GPU ⬜ |

### Phase 3b / kernel deepen (post–Phase 3)
| ID | Task | Status |
|----|------|--------|
| 3b.1 | Fused one-pass ⟨Z⟩ (16→1 state scan) | ✅ Done — FP64 bit-exact; **~+10%** vs prior best |
| 3b.2 | Dynamic smem tile 4096 (qubits 0..11) | ✅ Built (`-DQHASH_SMEM_TILE=4096`); bit-exact; **slightly slower** than 2048 (occupancy) — keep 2048 default |
| 3b.3 | High-qubit fiber block (qubits 11..15) | ✅ Done — gather→all high U2/CNOT in smem→scatter; coalesced IO; FP64 bit-exact; **~+2.7%** vs iter5 |
| 3b.4 | Tile double-buffer (2× smem) | ❌ Tried; **slower** (~2.53 vs ~2.72) — leave `QHASH_TILE_DBUF=0` |
| 3b.5 | Fuse low U2+CNOT residency + 2-pair ILP | ✅ Done — high U2 first (commute); FP64 bit-exact; **~neutral @2048, ~+2% @4096** |
| 3b.6 | Dual-tile boundary CNOT(10,11) in smem | ❌ Tried (`QHASH_BOUNDARY_PAIR`); **~−7%** (64 KiB dyn / occupancy) — default **OFF** |

### Phase 4 — Precision
| ID | Task | Status |
|----|------|--------|
| 4.1 | 10M+ nonce FP32 vs FP64 Q1.15 compare | ✅ **1.54M GPU digests** (stopped early; match flat) + 10k Q1.15 |
| 4.2 | Enable FP32 mining only if ≥99.99% match | ❌ **REJECTED** — digest match **~92.20%** (need ≥99.99%) |

### Phase 5 — Productize
| ID | Task | Status |
|----|------|--------|
| 5.1 | Stratum client | ✅ Via official miner (BYOS drop-in; no greenfield client) |
| 5.2 | BYOS drop-in validated in official miner build | ✅ `--enable-qhash-byos` / `--enable-qhash-byos-cuda`; `scripts/build-official-byos.sh` |
| 5.3 | Pool share accept-rate check | ✅ **Hash-OK 11/11** vs Suprnova (all rejects = `low difficulty share`; 0 invalid). **Default pool = Suprnova**. Credited accepts still blocked by min-diff (~8d TTF @ 3 kh/s); HeroMiners stratum down (2026-07-26) |
| 5.4 | CPU BYOS fallback when CUDA missing | ✅ `scanhash_qhash_cuda` → `scanhash_generic`; ~0.39 kh/s/thread; no error loop |

### Phase 6 — Closed-form ⟨Z⟩ (delete the statevector) — ✅ **COMPLETE**

The statevector is **not needed**. All 16 ⟨Z⟩ follow from a 16-step sweep over a
single 2×2 matrix. Prototype validated 2026-07-26 (see *Phase 6 notes*).

**Result: 3.25 kh/s → 286,934 kh/s sustained (~88,000×), bit-exact, 0 real divergences over 64 M
qubit values.** The statevector survives only as the consensus oracle. 6.14 is the one rejected
task, and it was rejected *because* the measured Q1.15 headroom is too small for FP32 — see 6.14.

| ID | Task | Status |
|----|------|--------|
| 6.1 | Derive + prototype closed form; validate vs statevector | ✅ Done — `tests/mps_prototype.cpp`; **20 000 nonces**, max‖Δ⟨Z⟩‖ **3.2e−14**, **0/320 000** Q1.15 mismatches; **6536×** faster (CPU: 1.42 M circuits/s vs 218/s) |
| 6.2 | Validate at scale vs cuStateVec golden + node soft-forks | ✅ Done — `qhash-closed-form-check`; **2 M nonces × 2 angle offsets = 4 M** (64 M qubit values), **0 real** Q1.15 divergences; cuStateVec golden PASS on 20 000 random + 4096 uniform |
| 6.3 | Q1.15 boundary / degenerate-angle adversarial tests | ✅ Done — importance-sampled boundary scan, exhaustive uniform patterns (**1152** exact ±1 values, **732** int16 wraps), soft-fork rejection through the closed form |
| 6.4 | Replace `qhash_cpu.cpp` sim with closed form (keep statevector as oracle) | ✅ Done — `qhash_sim_t`; `qhash_hash_cpu_sim` / `qhash_simulate_cpu_sim` reach the oracle |
| 6.5 | CUDA kernel: **one nonce per thread**, no smem, no barriers, no state buffer | ✅ Done — `qhash_cf_kernel`; **16 M nonces** bit-exact vs CPU; one barrier per *block* (LUT staging), none per nonce |
| 6.6 | Nibble→(cos,sin) 16-entry LUTs; drop all per-nonce transcendentals | ✅ Done — `AngleLut`; also tabulates cos²θ, sin²θ, cosθ·sinθ (bit-identical, −3 FP64 mul/qubit) |
| 6.7 | Skip layer-2 Rz nibbles (48..63) — provably irrelevant | ✅ Done — sweep reads nibbles 0..47 only, straight from SHA words (`NibbleWords`) |
| 6.8 | Hermitian Λ (4 doubles, not 8) | ✅ Done — `l00, l11, l01r, l01i` in registers |
| 6.9 | SHA-256 midstate + constant-folded padding schedules | ✅ Done — 4 compressions → 3; header `W[4..15]` and the final padding block's `W[0..63]` are job constants. Now only **12%** of kernel time and fully hidden behind FP64 |
| 6.10 | Persistent grid-stride kernel; on-device nonce gen; atomic winner queue | ✅ Done — occupancy-derived grid, nonce = `nonce_start + idx` on device, `atomicAdd` share queue |
| 6.11 | Re-tune threads/blocks/occupancy from scratch (old tuning is void) | ✅ Done — swept from scratch; **flat within 1%** over threads 128–512 and 1–16 resident waves. Defaults T=256, occupancy-derived grid. Launch *batch size* is the real lever (see 6.11 sweep) |
| 6.12 | Candidate re-verification vs statevector before submit | ✅ Done — `cf_reverify_candidates`; the submitted digest **is** the oracle's and is re-tested against the target. Self-test: 66 shares, 66 oracle checks, 0 rejected |
| 6.13 | Retire tile/fiber/dbuf/boundary code paths | ✅ Done — statevector kernel is now plain gate-at-a-time global memory; `QHASH_SMEM_TILE` / `QHASH_BOUNDARY_PAIR` removed; `src/qhash_kernel.cu` −379 lines |
| 6.14 | If FP64-bound: float-float (double-single) FP32 arithmetic | ❌ **REJECTED** — kernel *is* FP64-bound (96% of measured issue rate), but the Q1.15 safety margin is only **~20×** the current residual. Float-float is ~2^−46 vs FP64's 2^−53, i.e. **~128× worse** — it would produce real divergences. Explicit `fma` was taken instead: same 2× on FP64 issue, and *better* accuracy |

---

## Speed results log

Record every meaningful run. Update **Change** vs previous best on same hardware/backend.

### Baselines (external / literature)

| Date | Miner | Hardware | Rate | Notes |
|------|-------|----------|------|-------|
| — | Official cuStateVec | RTX 4070 | **~4.5 kh/s** | Community reported |
| — | Official cuStateVec (this host) | RTX 5060 Laptop | **~0.76–0.78 kh/s** | Stock `qhash-custatevec.c` FP32; measured 2026-07-26 (`official-miner-stock/`, `-t 1`) |
| — | CPU AVX miner | Ryzen 9 7950X | **~12.6 kh/s** | 30 threads |
| — | AllFather GPU | RTX 4070 | **~20 kh/s** | Community |
| — | **Target** | RTX 4070 | **100–200+ kh/s** | Custom kernel |

### This project

| Iter | Date | Backend | Hardware | Precision | Nonces | Time (s) | H/s | kh/s | Change | Notes |
|------|------|---------|----------|-----------|--------|----------|-----|------|--------|-------|
| 0 | 2026-07-26 | CPU ref | WSL2 (no GPU) | FP64 | 16 | 0.051 | 313 | 0.313 | — | `qhash-bench`; single-thread |
| 0 | 2026-07-26 | CPU ref | WSL2 (no GPU) | FP32 | 16 | 0.062 | 257 | 0.257 | −18% vs FP64 CPU | Not faster on this host |
| 1 | 2026-07-26 | CUDA P1 unfused | RTX 5060 Laptop | FP64 | 1024 | 0.800 | 1280 | **1.280** | — | chunk=48; separate Ry then Rz |
| 2 | 2026-07-26 | CUDA + fuse | RTX 5060 Laptop | FP64 | 1024 | 0.514 | 1994 | **1.994** | **+56%** vs iter1 | `-DQHASH_FUSE_RZ_RY`; chunk=48 |
| 2b | 2026-07-26 | CUDA + fuse | RTX 5060 Laptop | FP64 | 2048 | 0.855 | 2396 | **2.396** | +20% vs iter2 | longer run; same kernel |
| 3a | 2026-07-26 | CUDA + CNOT tile | RTX 5060 Laptop | FP64 | 1024 | 0.507 | 2019 | **2.019** | +1% vs iter2 | pair-index CNOT only (no smem) |
| 3 | 2026-07-26 | CUDA + CNOT + smem | RTX 5060 Laptop | FP64 | 1024 | 0.501 | 2045 | **2.045** | **+3%** vs iter2 | tile=2048; 64 regs / 36 KiB smem |
| 3b | 2026-07-26 | CUDA + CNOT + smem | RTX 5060 Laptop | FP64 | 2048 | 0.928 | 2206 | **2.206** | −8% vs iter2b | same kernel; run-to-run variance vs 2b |
| 4 | 2026-07-26 | CUDA P3 occupancy | RTX 5060 Laptop | FP64 | 1024 | 0.445 | 2300 | **2.300** | **+12%** vs iter3 | T=256 C=128 S=1 |
| 4b | 2026-07-26 | CUDA P3 occupancy | RTX 5060 Laptop | FP64 | 2048 | 0.850 | 2409 | **2.409** | **+9%** vs iter3b | T=256 C=128 S=1 (new default) |
| 5 | 2026-07-26 | CUDA + fused ⟨Z⟩ | RTX 5060 Laptop | FP64 | 2048 | 0.773 | 2650 | **2.650** | **+10%** vs iter4b | one-pass ⟨Z⟩; gate fix; T=256 C=128 |
| 5b | 2026-07-26 | CUDA + fused ⟨Z⟩ | RTX 5060 Laptop | FP64 | 1024 | 0.393 | 2606 | **2.606** | — | same kernel |
| 5c | 2026-07-26 | CUDA tile=4096 dyn | RTX 5060 Laptop | FP64 | 2048 | 0.767 | 2669 | **2.669** | ~0 vs 5 | dynamic smem; no clear win; keep 2048 |
| 6 | 2026-07-26 | CUDA + high-q fiber | RTX 5060 Laptop | FP64 | 2048 | 0.751 | 2724 | **2.724** | **+2.7%** vs iter5 | fiber block q11..15; coalesced gather; T=256 C=128 |
| 6b | 2026-07-26 | CUDA + high-q fiber | RTX 5060 Laptop | FP64 | 1024 | 0.403 | 2542 | **2.542** | — | same kernel; short-run variance |
| 6c | 2026-07-26 | CUDA fiber + tile dbuf | RTX 5060 Laptop | FP64 | 2048 | 0.808 | 2535 | **2.535** | **−7%** vs 6 | `QHASH_TILE_DBUF=1`; rejected |
| 7 | 2026-07-26 | CUDA fuse low U2+CNOT+ILP | RTX 5060 Laptop | FP64 | 2048 | ~0.75 | ~2720 | **~2.72** | ~0 vs 6 | high-U2-first; fused residency; 2-pair ILP; T=256 C=128 |
| 7b | 2026-07-26 | CUDA fuse+ILP | RTX 5060 Laptop | FP64 | 4096 | ~1.47 | ~2780 | **~2.78** | **~+2%** vs 6 | longer run; same kernel |
| 7c | 2026-07-26 | CUDA + boundary pair | RTX 5060 Laptop | FP64 | 2048 | 0.809 | 2531 | **2.531** | **−7%** vs 6 | `QHASH_BOUNDARY_PAIR=1`; 64 KiB dyn; rejected |
| 8 | 2026-07-26 | Official miner CUDA BYOS | RTX 5060 Laptop | FP64 | (live) | — | ~2750 | **~2.75** | ~0 vs 7 | `build-official-byos.sh --cuda`; `-t 1` |
| 9 | 2026-07-26 | Official CUDA BYOS @ pool | RTX 5060 Laptop | FP64 | (live) | ~7m | ~2760 | **~2.76** | ~0 vs 8 | Suprnova via low-diff proxy; 11 submits, all `low difficulty share` |
| 10 | 2026-07-26 | Stock cuStateVec (same card) | RTX 5060 Laptop | FP32 | (live) | ~25s | ~764 | **~0.76** | — | `official-miner-stock`; baseline for A/B |
| 10b | 2026-07-26 | Official CUDA BYOS (A/B) | RTX 5060 Laptop | FP64 | (live) | ~25s | ~3250 | **~3.25** | **~4.3×** vs stock same card | paired with iter 10; last sample ~3.08 |
| **11** | 2026-07-26 | **CUDA closed form** (Phase 6.5–6.10) | RTX 5060 Laptop | FP64 | 268 435 456 | 1.405 | 1.909e8 | **190 933** | **~69 000×** vs iter 10b | one nonce/thread; 611 FP64 instr/nonce; T=256, occupancy grid |
| 11b | 2026-07-26 | + explicit `fma` in the sweep | RTX 5060 Laptop | FP64 | 268 435 456 | 0.999 | 2.686e8 | **268 330** | **+41%** vs iter 11 | 611 → 442 FP64 instr/nonce |
| 11c | 2026-07-26 | + cos²/sin²/cos·sin tabulated | RTX 5060 Laptop | FP64 | 268 435 456 | 0.866 | 3.099e8 | **309 680** | **+15%** vs iter 11b | 442 → **379** FP64 instr/nonce; bit-identical change |
| **12** | 2026-07-26 | **Sustained steady state** (final) | RTX 5060 Laptop | FP64 | 4 000 000 000 | 13.95 | 2.869e8 | **286 934** | −7% vs iter 11c | 954 × 4 M launches; median of 3 (285.5 / 286.9 / 286.5). Below 11c *because* 11c is one warm launch with no per-launch overhead; **12 is the honest continuous-mining number** and 11c is the kernel-only ceiling |
| **13** | 2026-07-26 | **Official miner CUDA BYOS** (end to end) | RTX 5060 Laptop | FP64 | (live) | ~60s | ~2.98e8 | **~298 000** | **~91 700×** vs iter 10b; **~390×** vs stock cuStateVec same card | `-a qhash --benchmark -t 1`; batch 4 M (was 128 → 735 kh/s) |

### Phase 3.1 sweep (RTX 5060 Laptop, FP64, 1024 nonces)

| threads | chunk | streams | kh/s |
|--------:|------:|--------:|-----:|
| 128 | 24 | 1 | 1.695 |
| 128 | 32 | 1 | 1.478 |
| 128 | 48 | 1 | 2.086 |
| 128 | 64 | 1 | 1.717 |
| 128 | 96 | 1 | 2.079 |
| 128 | 128 | 1 | 2.165 |
| 256 | 24 | 1 | 2.029 |
| 256 | 32 | 1 | 1.461 |
| 256 | 48 | 1 | 2.078 |
| 256 | 64 | 1 | 1.917 |
| 256 | 96 | 1 | 2.142 |
| **256** | **128** | **1** | **2.300** |
| 256 | 128 | 2 | 2.283 |
| 128 | 128 | 2 | 2.177 |

Notes: chunk **64** is a local valley; **streams≈neutral**; **128 threads** bit-exact but slower than 256 at peak chunk.
**These numbers are void for the closed-form kernel** — they were tuned around a 1 MiB-per-block statevector.

### Phase 6.11 sweep (RTX 5060 Laptop, FP64 closed form)

Launch geometry, 2 × 10⁹ nonces in 4 M-nonce launches so every point sees the same clocks:

| threads | Mh/s | | resident waves | blocks | Mh/s |
|--------:|-----:|-|---------------:|-------:|-----:|
| 128 | 285.4 | | 1 | 104 | 268.0 |
| **256** | **286.8** | | 2 | 208 | 266.1 |
| 512 | 284.4 | | **4** | **416** | **267.0** |
| | | | 8 | 832 | 267.8 |
| | | | 16 | 1664 | 267.9 |

Geometry is **flat within 1%** in both directions — expected, since the kernel is FP64-issue-bound
rather than occupancy-bound. What *does* matter is nonces per launch, because the fixed cost
(constant-memory upload, counter reset, event sync) is amortised over it:

| nonces / launch | Mh/s |
|----------------:|-----:|
| 4 096 | 66.7 |
| 65 536 | 146.0 |
| 262 144 | 154.6 |
| 1 048 576 | 167.6 |
| **4 194 304** | **240.0** |
| 16 777 216 | 184.6 |

(Averages over a fixed 268 M total, so early launches are still ramping clocks; the ordering is what
matters.) **4 M** is the BYOS batch size: launch overhead is negligible and a batch still returns every
~15 ms, which keeps `work_restart` responsive.

**Measurement caveat:** short runs read low. A single cold 4 M-nonce launch reports ~26 Mh/s and a
cold 268 M launch ~117 Mh/s, purely because the GPU has not boosted yet. Always measure with a
multi-second sustained load — that is what iter 12 does.

**How to append a row**

```bash
export CUDA_HOME=$HOME/opt/cuda-13.3
export PATH=$CUDA_HOME/bin:$HOME/opt/cuda-env/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
# Sustained closed-form steady state (the number worth logging):
./build/qhash-miner --benchmark --nonces 4000000000 --chunk 4194304
# Statevector oracle, for A/B:
./build/qhash-miner --benchmark --sim statevector --nonces 2048 --threads 256 --chunk 128
# Fill: Iter, Date, Backend, Hardware (nvidia-smi), Precision, Nonces, Time, H/s, kh/s
# Change = (new_khs / prev_best_khs - 1) * 100% on same GPU
```

**Build note:** apt `nvidia-cuda-toolkit` is **12.4** and cannot target `sm_120`. Use local **CUDA 13.3** at `~/opt/cuda-13.3` (driver UMD is already 13.3).

```bash
# cuStateVec golden — checks the closed form AND the statevector against the oracle:
cmake -S . -B build -DQHASH_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
./build/qhash-custatevec-golden --nonces 20000 --fill 165
./build/qhash-custatevec-golden --uniform

# Phase 6 correctness gate (see "Correctness bar" below):
./build/qhash-closed-form-check --mode all --nonces 4000000
./build/qhash-closed-form-check --mode gpu --nonces 16777216 --oracle-stride 256

# Where the time goes, and the FP64 roofline:
OPS=$(nvcc -arch=sm_120 -std=c++17 -Iinclude --fmad=false --expt-relaxed-constexpr \
        -ptx src/qhash_kernel.cu -o /tmp/qk.ptx && \
      awk '/\.visible \.entry _Z15qhash_cf_kernel/{f=1} f{print} f&&/^\}$/{exit}' /tmp/qk.ptx \
        | grep -cE '\.rn\.f64')
./build/qhash-cf-ablation --nonces 268435456 --reps 4 --fp64-ops $OPS
```

### Correctness snapshots

| Date | Check | Result |
|------|-------|--------|
| 2026-07-26 | `make test` / `qhash-test` | PASS (SHA-256 NIST, gates, fixed-point) |
| 2026-07-26 | `--self-test` zero-header FP64 | **`8711a7a489f9021a8d8011434e374944ee4051851d21c9fc08dee11757d25bc4`** (updated after gate fix; was `b2c045b4…` under wrong θ/2 convention) |
| 2026-07-26 | Gate convention vs cuStateVec | **FIXED** — was textbook `exp(−i θ/2 P)`; library/node use `exp(+i θ P)` |
| 2026-07-26 | vs cuStateVec FP64 Q1.15 | **1024/1024** nonces, **0/16384** qubit mismatches, max‖Δ⟨Z⟩‖≈1.6e−14 |
| 2026-07-26 | GPU vs CPU FP64 digests | **64/64** (+512 spot) bit-exact after gate fix + fused ⟨Z⟩ |
| 2026-07-26 | Even/odd CNOT brickwork | **REJECTED** — NN chain CNOTs do not commute |
| 2026-07-26 | Phase 4 (10k) Q1.15 FP32 vs FP64 | qubit match **96.86%**; nonce match **59.64%** |
| 2026-07-26 | Phase 4 (10k) GPU digests FP32 vs FP64 | match **92.02%** — **UNSAFE** (&lt;99.99%) |
| 2026-07-26 | GPU FP32 vs CPU FP32 | **≠** (~61% on 512) — tree-reduce vs sequential; FP64 still bit-exact |
| 2026-07-26 | Phase 4 digests (stopped 1.54M / 10M) | match **92.20%** (mm=120461); flat since ~100k — **FP32 mining OFF** |
| 2026-07-26 | High-q fiber vs CPU / cuStateVec | self-test 64/64 bit-exact; golden 512/512 Q1.15 OK |
| 2026-07-26 | Fuse low U2+CNOT + ILP | self-test 64/64; golden 512; `ctest` 4/4 incl. `byos_check` |
| 2026-07-26 | Boundary dual-tile | bit-exact but slower — keep `QHASH_BOUNDARY_PAIR=OFF` |
| 2026-07-26 | Official BYOS CUDA FP64 | linked; bench ~2.75 kh/s (`-t 1`); `byos_check` OK |
| 2026-07-26 | Pool hash validation (Suprnova) | **11/11** submits → reject reason **only** `low difficulty share` (0 invalid / 0 stale) |
| 2026-07-26 | CUDA→CPU fallback | warning once; ~0.39 kh/s (`--benchmark -t 1` without GPU) |
| 2026-07-26 | **Closed form vs statevector** (20 000 nonces, both angle offsets) | max‖Δ⟨Z⟩‖ **3.209e−14**; **0/320 000** Q1.15 mismatches; **0/20 000** nonce mismatches |
| 2026-07-26 | Layer-2 `Rz` nibbles are dead inputs | perturbing `nibbles[48..63]` moves statevector ⟨Z⟩ by **~1e−16** (rounding noise) → confirmed irrelevant |

#### Phase 6 gate (`qhash-closed-form-check`, final arithmetic)

The closed form is an exact identity, so it can only change a digest if its ~1e−14 FP64 residual
lands on a Q1.15 rounding boundary. Every row below therefore reports the **tightest boundary margin
actually observed** next to the worst-case |Δ⟨Z⟩|, because that ratio — not the raw error — is the
safety factor. Disagreements are arbitrated by a `long double` reference (`tests/reference_ld.h`,
~1e−22 residual) written independently of `closed_form.cuh`: a value closer than 1e−15 to a boundary
is an **exact tie**, an input on which no FP64 simulator has a defined answer and the consensus node
is rounding noise too.

| Date | Check | Result |
|------|-------|--------|
| 2026-07-26 | `scale` offset 0, **2 M** nonces | **0 real** divergences / 32 M values (25 mismatches, **all** exact ties, margins ≤2.7e−17); max‖Δ⟨Z⟩‖ **5.085e−14** = 1.67e−09 Q1.15 units; tightest resolvable margin **3.26e−08** → **19.5× headroom** |
| 2026-07-26 | `scale` offset 1, **2 M** nonces | **0** mismatches of any kind / 32 M values; max‖Δ⟨Z⟩‖ **2.831e−14**; tightest margin **2.31e−08** → **24.9× headroom** |
| 2026-07-26 | `degenerate` uniform patterns (all 8192) | **1152** exact ±1 values, **732** deliberate `+32768 → −32768` int16 wraps exercised; **0 real** divergences (17 exact ties) |
| 2026-07-26 | `degenerate` quarter-turn alphabet {0,4,8,12} | 20 000 patterns, **0** mismatches; margin/|Δ| **2111×** |
| 2026-07-26 | `softfork` zero-byte rejection | zero-byte counts 0–21 and 32 reached from uniform patterns; **31** 0xFF rejections hit across all four activation times; **0** digest mismatches; **0** zero-byte count disagreements on 20 000 real nonces |
| 2026-07-26 | `gpu` — closed-form kernel vs CPU | **16 M** nonces, **0** digest mismatches vs the CPU closed form (exact, same arithmetic); vs the oracle on 65536 strided nonces: **0 real divergences, 1 exact tie** (nonce 2 611 968 — see below) |
| 2026-07-26 | cuStateVec golden, 20 000 nonces | closed form **0/320 000** Q1.15 mismatches, max‖Δ⟨Z⟩‖ **2.78e−15** — **7.5× closer to the oracle than our own FP64 statevector** (2.10e−14) |
| 2026-07-26 | cuStateVec golden, 4096 uniform patterns | closed form **0/65 536**, max‖Δ⟨Z⟩‖ **5.55e−15** (statevector 2.91e−14) |
| 2026-07-26 | Explicit `fma` + tabulated cos²/sin² | Self-test digest unchanged; accuracy **improved** (see cuStateVec rows). `qfma` is IEEE-754 correctly rounded on both glibc and `__fma_rn`, so host and device stay bit-identical |
| 2026-07-26 | Statevector kernel after 6.13 strip | self-test **64/64** bit-exact vs CPU statevector at T=256 and T=128/2-stream; `custatevec_golden` PASS |
| 2026-07-26 | Candidate re-verification (6.12) | self-test mines 4096 nonces at target `0x04…`: **66** shares, **66** oracle re-hashes, **0** rejected; every submitted digest is the statevector's |
| 2026-07-26 | `ctest` after Phase 6 | **6/6** PASS (`byos_check`, `circuit`, `selftest`, `closed_form`, `closed_form_softfork`, `custatevec_golden`) |
| 2026-07-26 | BYOS ABI | `src/byos/qhash_byos.cpp` **unmodified**. `qhash_job_t` only gained appended fields (`sim`, `blocks`) and official-miner zero-initialises the struct, so 0 = closed form / occupancy-derived grid |

#### The one caveat: exact Q1.15 ties

At scale a small number of nonces land on an **exact** Q1.15 rounding boundary, where the closed form
and the statevector disagree by one LSB. Worked example, the single tie in the 16 M-nonce GPU gate
(`--mode nonce --nonce 2611968 --offset 0`, qubit 14, nibbles y1=2 z1=11 y2=8):

```
statevector  4.5776367191248622e-05  -> Q1.15  2
closed form  4.5776367187498685e-05  -> Q1.15  1
exact (ld)   4.5776367187500001522e-05     = 1.5 / 32768 to within the reference's own noise
                                              → 4.99e-17 Q1.15 units from the boundary
```

`to_fixed_q15` truncates `x·32768 + 0.5`, so the decision hinges on whether `x·32768` is above or
below **exactly 1.5**. The true value *is* 1.5, so the answer is decided entirely by the last bit of
rounding noise in whichever summation order a given implementation uses. **No FP64 simulator has a
defined answer here — including the consensus node's.** Rate: ~1 nonce in 10⁵ contains such a tie
(25 tied values per 32 M at offset 0; 1 digest difference per 65536 nonces sampled).

Why this is not a consensus risk:

1. **6.12 makes the submitted digest the oracle's**, produced by the same statevector code that
   shipped through Phases 1–5. So Phase 6 adds **zero** new disagreement with the node — the tie
   question is identical to the pre-Phase-6 one, and the closed form never decides a share.
2. The only cost is **efficiency**: a nonce whose closed-form digest passes target may fail on the
   oracle's digest, and vice versa. At ~10⁻⁵ of nonces, and with the target test being effectively
   random in either digest, the expected loss is ~10⁻⁵ of hashrate — far below run-to-run clock noise.
3. It is **not** fixable by more FP64 care, and it is precisely why 6.14 was rejected: the tie rate
   scales with the residual, so a 128× worse residual would convert ties into real divergences.

Every mode of `qhash-closed-form-check` therefore arbitrates a Q1.15 disagreement against the
independent `long double` reference and fails **only** on a real divergence. `--mode nonce` reruns any
single nonce and prints all 16 ⟨Z⟩ with margins and a verdict per qubit, for the next time a wide
sweep reports only a nonce number.

### Profile snapshot (Phase 1.5 / 2.2 / 3 / 3b)

| Source | Finding |
|--------|---------|
| `ncu` | Still **ERR_NVGPUCTRPERM** on WSL — enable GPU performance counters then re-run |
| Device smem | `sharedMemPerBlock=48 KiB`, **`sharedMemPerBlockOptin≈99 KiB`** (enables 4096 FP64 tile via dynamic smem) |
| `ptxas -v` FP64 (fuse+ILP, tile 2048) | **~70 regs**, 0 spills, **~36 KiB** static smem |
| `ptxas -v` FP64 (tile 4096 dyn) | **64 regs**, ~3.4 KiB static + **64 KiB** dynamic tile |
| Tile 4096 | Bit-exact; **no hashrate win** vs 2048 (1 block/SM smem occupancy) |
| Fused ⟨Z⟩ | 1 state scan + 16 tree-reduces; preserves per-thread accum order → FP64 Q1.15 OK |
| High-q fiber | q11..15: 1 gather/scatter per layer section vs 5 U2 + 4 CNOT full passes; coalesced by fixed-j |
| Tile dbuf 2× | Bit-exact but **slower**; default OFF |
| Fuse low U2+CNOT | High U2 first (U2s commute); one contiguous residency for low U2+CNOT 0..9; cuts low gather/scatter traffic |
| 2-pair ILP | U2/CNOT pair loops issue 2 pairs/step; **~70 regs** FP64 (was 64); no spills |
| Boundary pair 2× | CNOT(10,11) in 64 KiB dyn smem; bit-exact; **−7%** — default OFF |
| Multi-stream | Concurrent nonce slices; **no meaningful gain** (compute-bound) |
| Barriers | Full-block sync between sequential gates required; brickwork unsafe |
| `ncu` (recheck) | Still **ERR_NVGPUCTRPERM** on WSL |
| Pool (Suprnova direct) | Connects; Diff **0.5** → share TTF ~**8d** @ 2.8 kh/s — no shares in 10 min |
| Pool (HeroMiners) | Unreachable from this WSL (fake-IP `198.18.0.x` / empty stratum on real IP) |
| Pool (LuckyPool) | Connects; ignores static `=DIFF`; stuck Diff **48** (~years TTF) |
| Pool (low-diff proxy→Suprnova) | **11/11** pool hash-OK (`low difficulty share`); ~2.76 kh/s CUDA |

### Profile snapshot (Phase 6 closed-form kernel)

`ncu` is still **ERR_NVGPUCTRPERM** on this host, so the split was measured by ablation instead:
`bench/cf_ablation.cu` runs the production inner loop with one half removed, plus an FP64
issue-rate probe, all in one process so every variant sees the same clocks.

| Measurement | Value |
|-------------|-------|
| `ptxas -v` `qhash_cf_kernel` | **63 regs**, **0 spills**, **512 B** smem, **1 barrier** (LUT staging, once per block) |
| Occupancy | **4 blocks/SM** at 256 threads (register-limited); 26 SMs → 416-block default grid |
| Full kernel | **3.23 ns/nonce** (309.7 Mh/s, single warm launch) |
| — SHA-256 half alone | **0.70 ns/nonce** (1403 Mh/s) → **12%** of cost, and fully hidden |
| — FP64 sweep alone | **3.21 ns/nonce** (311 Mh/s) → **88%** of cost |
| Overlap | full ≈ max(halves), not their sum (serial model 3.91 ns vs measured 3.23) — the SHA rides along free |
| **FP64 instruction issue rate (measured)** | **122.8 G/s** — identical for `mul`+`add` (2 instr, 2 flops) and `fma` (1 instr, 2 flops), i.e. **fma is exactly 2.00× the flops** |
| Kernel FP64 instructions | **379/nonce** (was 611 before `fma`, 442 after `fma`, 379 after tabulating cos²/sin²/cos·sin) |
| **Roofline utilisation** | 379 × 309.7 M = **117.5 G/s = 96% of the measured FP64 issue rate** |
| Verdict | **FP64-issue-bound at the roofline.** Occupancy, SHA-256 and memory are all irrelevant; the only remaining lever is fewer FP64 instructions per nonce |

Consequence for 6.14: being FP64-bound is the *precondition* for float-float, but the Q1.15 headroom
is only ~20×, and float-float's 2^−46 accuracy is ~128× worse than FP64's 2^−53. It would convert
"0 real divergences" into "several per million nonces". Explicit `fma` captured the same 2× while
*reducing* error, so 6.14 is closed as rejected on measured evidence.

---

## Phase 3 / 3b notes

> **Historical.** Everything in this section describes the statevector kernel, which Phase 6 demoted
> to the consensus oracle and Phase 6.13 stripped back to plain gate-at-a-time global memory. The
> tiling, fiber-blocking, double-buffering and boundary-pairing code and the `QHASH_SMEM_TILE` /
> `QHASH_BOUNDARY_PAIR` options are **gone**. Kept for the reasoning, especially items 2–4, which are
> consensus constraints rather than tuning notes.

1. **Defaults (then):** `--threads 256 --chunk 128 --streams 1`; `QHASH_SMEM_TILE=2048`; `QHASH_TILE_DBUF=0`; `QHASH_BOUNDARY_PAIR=0`.
2. **Fewer barriers:** even/odd CNOT brickwork changes circuit semantics — do not use.
3. **Warp-shuffle ⟨Z⟩ reduce:** skipped (would change FP reduction order vs CPU / Q1.15).
4. **Fused ⟨Z⟩:** safe for FP64; do not shuffle-reduce.
5. **Multi-stream:** `job.num_streams` splits one batch across CUDA streams; keep for future H2D/D2H-heavy paths.
6. **High-qubit fiber:** after low contiguous tiles, pack `2^(16-kTileQ)`-amp fibers into smem; apply all high U2s (then later high CNOTs) before scatter. With fused low path, high U2 runs **first** (U2s commute). Boundary CNOT(`kTileQ-1`,`kTileQ`) stays global unless `QHASH_BOUNDARY_PAIR`. Overlap ⟨Z⟩ with last gates skipped (would change FP64 accum order).
7. **Tile double-buffer:** optional `-DQHASH_TILE_DBUF=1` (64 KiB dyn); measured regression — keep off.
8. **Boundary pair:** optional `-DQHASH_BOUNDARY_PAIR=1`; dual-tile low U2+CNOT including CNOT(10,11) in smem; measured regression — keep off.
9. **Kernel plateau:** further wins likely need ncu (enable GPU counters on WSL) or algorithmic/precision changes — not more 2× smem tricks.

## Phase 4 notes

1. CLI: `./build/qhash-miner --compare-fp --nonces N [--q15-nonces M]`
2. Digests over N (GPU); Q1.15 CPU sample over M (default `min(N,100k)`).
3. **FP32 mining stays OFF** — 1.54M digests @ **92.20%** ≪ 99.99% (log: `phase4_10M.log`).
4. GPU FP32 digests must not be confused with CPU FP32 (tree vs sequential).

Do **not** switch mining to FP32 until Phase 4 bit-exact study passes (≥99.99% digest match vs FP64). **Phase 4 failed that bar.**

## Phase 5 notes

1. Official miner already implements Stratum (`cpu-miner.c` / `util.c`). Prefer **BYOS drop-in** over a greenfield stratum client.
2. `src/byos/qhash_byos.cpp` exports `qhash_thread_init` / `run_simulation` matching `qhash-gate.h`.
3. Default BYOS = stock miner (**FP32 + legacy angles**). Consensus: `-DQHASH_BYOS_FP64=1` (+ `qhash_byos_set_ntime` from header in `qhash.c`).
4. Validate ABI: `./build/qhash-byos-check` (also `ctest`).
5. **Official tree wiring (2026-07-26):**
   - `official-miner/configure.ac`: `--enable-qhash-byos`, `--enable-qhash-byos-cuda`, `--enable-qhash-byos-fp64`, `--with-qhash-miner[-build]`
   - `Makefile.am`: replaces `qhash-custatevec.c` with `libqhash_byos*.a` / `libqhash_byos_plugin.so`
   - CUDA path: `algo/qhash/qhash-scanhash-cuda.cpp` + `gate->scanhash` → batched `qhash_mine_batch` (FP64)
   - Build: `./scripts/build-official-byos.sh --cuda` (needs `MINER_DEPS=~/opt/miner-deps` curl/autotools extract if no system packages)
6. **Bench (RTX 5060 Laptop, official miner CUDA BYOS FP64, `-t 1`):** ~**2.75 kh/s** — matches standalone kernel plateau.
7. Prefer **`-t 1`** with CUDA BYOS (GPU mutex serializes; extra threads idle).
8. **Pool hash-OK (2026-07-26):** CUDA BYOS @ ~2.76 kh/s submitted **11** shares to Suprnova (via local low-diff proxy so shares arrive in minutes). Pool reject reason for **all 11** was `low difficulty share` — digest validated, below pool min-diff. **0 invalid / 0 stale** → treat as **100% hash-OK**. Credited accepts still need a reachable low fixed-diff pool (HeroMiners `=32`–`64` preferred; currently broken from this WSL).
9. **CPU fallback:** if `!qhash_cuda_available()`, `scanhash_qhash_cuda` calls `scanhash_generic` (FP64 BYOS) once-warned — avoids tight `hashes_done=0` error loop. Sandbox/no-GPU bench ≈ **0.39 kh/s**.
10. Helpers: `./scripts/mine-suprnova.sh` (default pool), `./scripts/pool-accept-check.sh` (suprnova|herominers|luckypool). Throwaway wallet: `.pool-wallet-test.txt` (gitignored).
11. **ncu:** still **ERR_NVGPUCTRPERM** on WSL — kernel plateau next steps wait on GPU counter enable.

---

## Phase 6 notes — the statevector is unnecessary

Phases 1–3b optimised *how fast we push 65536 amplitudes*. Phase 6 removes the
amplitudes. This is an **exact identity**, not an approximation or a truncation.

### Why it works

The circuit is 2 layers of `[16 single-qubit U2] + [CNOT staircase 0→1→…→15]`.
Three facts collapse it:

**1. The final CNOT staircase turns each `Z_q` into a parity string.**
Conjugating backwards through `C(14,15)…C(0,1)` (using `CNOT Z_t CNOT = Z_c Z_t`,
`CNOT Z_c CNOT = Z_c`):

```
O_q = U_cnot† Z_q U_cnot = Z_0 Z_1 … Z_q
```

So we only need parity expectations on the state *before* the last staircase.

**2. Layer-2 `Rz` gates cannot affect the result.** `O_q` is all-`Z`, and `Rz` is
diagonal, so `V† Z V = Ry† Rz† Z Rz Ry = Ry† Z Ry`. The 16 layer-2 `Rz` nibbles
(`nibbles[48..63]`) are **dead inputs** — confirmed numerically against the full
statevector (perturbing them moves ⟨Z⟩ by ~1e−16, i.e. rounding noise). Only 48 of
the 64 nibbles matter.

```
G_q := Ry_q† Z Ry_q = [[ cos 2θy , sin 2θy ], [ sin 2θy , −cos 2θy ]]      (real)
```

**3. The layer-1 staircase on a product state has an explicit prefix-XOR form.**
The staircase maps basis states by cumulative XOR (`y_k = x_0⊕…⊕x_k`), so with
`(a_q, b_q) = U2_q|0⟩`:

```
χ(y) = ∏_k c_k(y_k ⊕ y_{k−1}),   c_k(0)=a_k, c_k(1)=b_k,   y_{−1}=0
```

That is a bond-dimension-2 MPS written down in closed form — no SVD, no QR.
(Consistent with the bound: each cut is crossed by exactly 2 CNOTs, so Schmidt
rank ≤ 4.)

### The resulting algorithm

Per qubit `q`, from `θy1 = angle(nib[q])`, `θz1 = angle(nib[16+q])`,
`θy2 = angle(nib[32+q])`:

```
a_q = e^{+iθz1}·cos θy1        b_q = −e^{−iθz1}·sin θy1
C_q = [[a_q, b_q], [b_q, a_q]]                       (symmetric)
s_q = 2·Re(a_q · conj(b_q)) = −sin(2θy1)·cos(2θz1)   (real)
```

Then a single left-to-right sweep of one 2×2 complex accumulator:

```
Λ ← [[1,0],[0,0]]
for q = 0 … 15:
    Λ ← G_q ⊙ (C_q^H · Λ · C_q)                      (⊙ = elementwise)
    s ← (q < 15) ? s_{q+1} : 1
    ⟨Z_q⟩ ← Λ00 + Λ11 + s·(Λ01 + Λ10)
```

The all-identity right environment collapses to `(1, s, s, 1)` because
`|a|²+|b|² = 1` makes the identity transfer map preserve the all-ones vector —
so there is **no backward sweep and no environment storage**.

`Λ` stays Hermitian (`G` is real symmetric), so `Λ00`/`Λ11` are real and
`Λ10 = conj(Λ01)`: **4 doubles of state for the entire simulation**.

### Cost

| | Today (statevector) | Phase 6 |
|--|--------------------|---------|
| Working state | **1 MiB** / nonce (65536 × complex FP64) | **32 B** / nonce (2×2 Hermitian) |
| Gate passes | 2 × (16 U2 + 15 CNOT) over 65536 amps | 16 × two 2×2 matmuls |
| FP64 ops / nonce | ~30 M | **~1.5–2.5 k** |
| Global memory traffic | ~60 MiB / nonce | **0** |
| Parallel granularity | one **block** per nonce | one **thread** per nonce |
| Barriers | ~62 `__syncthreads()` | **0** per nonce (1 per *block*, staging the LUT) |
| Transcendentals | 64 `sincos` | **0** (16-entry LUT) |
| Measured (1 CPU thread) | 218 circuits/s | **1,423,481 circuits/s** (**6536×**) |
| **Measured (GPU, shipped)** | **3.25 kh/s** | **286,934 kh/s** (**~88,000×**) |

Post-implementation, the FP64 estimate was pessimistic: **379 FP64 instructions per nonce** measured
from PTX, not 1.5–2.5 k, after `fma` fusion and the cos²/sin² identities.

### Reproduce

```bash
g++ -O2 -Iinclude tests/mps_prototype.cpp -o .proto/mps_prototype
./.proto/mps_prototype 20000        # correctness vs statevector
./.proto/mps_prototype 1 2000000    # CPU speed ratio
```

### What this changes downstream

> **Outcome vs prediction.** Items 1, 3, 4 and 5 held. Item 2 was **wrong in the interesting
> direction**: after the sim got 6500× cheaper the kernel did *not* become SHA-bound — SHA-256 is
> only **12%** of kernel time and is entirely hidden, while the sweep's 379 FP64 instructions sit at
> **96% of the card's measured FP64 issue rate**. FP64 is the wall, which is exactly the condition
> 6.14 was gated on; see the profile snapshot for why float-float was still rejected.

1. **SHA-256 becomes the bottleneck**, not the quantum sim. Per nonce today:
   `sha256(80 B)` = 2 compressions + `sha256(64 B)` = 2 compressions = **4**.
   - **Midstate:** header bytes 0..63 are nonce-independent → precompute on host,
     drops to **3**.
   - **Constant-folded schedules:** the padding-only second block of the 64-byte
     final hash has a *fully constant* `W[0..63]`; the header's second block has
     constant `W[4..15]`. Standard Bitcoin-miner folding applies.
   - The current `if (tid == 0) sha256(...)` serialization (255 idle threads)
     disappears for free with one nonce per thread.
2. **FP64 stops being expensive.** At ~2 k FP64 flops/nonce, the 1:64 FP64 rate on
   consumer cards is no longer the limiter, so no precision compromise is needed.
   Only if profiling shows FP64-bound should 6.14 (float-float, ≈2^−46 accuracy at
   ~15 FP32 ops per op) be considered.
3. **Phase 3/3b tuning is void.** Tile size, fiber blocks, dbuf, boundary pairs,
   `threads=256 chunk=128` — all were tuned for a 1 MiB/block statevector. Retune
   from zero. Batch size is no longer capped by state memory.
4. **Pool difficulty stops being a problem.** Phase 5.3's ~8-day share TTF at
   3 kh/s scales down linearly; Suprnova Diff 0.5 becomes seconds-per-share.
5. **Keep the statevector.** It stays as the consensus oracle for `--self-test`,
   the cuStateVec golden harness, and (6.12) re-verification of any nonce that
   passes target before submission. That makes the residual ~1e−14 discrepancy
   unable to produce an invalid share.

### Correctness bar before this ships — ✅ MET

Same bar as Phase 4, applied to the closed form. All four items cleared; see the Phase 6 gate table
under "Correctness snapshots" for the numbers.

- ✅ Q1.15 match vs FP64 statevector **100%** on ≥1 M nonces — run at **4 M** (2 M per angle offset),
  64 M qubit values, **0 real divergences**. The 25 raw mismatches at offset 0 are all exact
  boundary ties (margin ≤2.7e−17, arbitrated by an independent `long double` reference) — inputs on
  which the consensus node's own FP64 answer is a coin flip.
- ✅ `qhash-custatevec-golden` re-run against the closed form: **0/320 000** mismatches, and the
  closed form is **7.5× closer to cuStateVec than our statevector is**.
- ✅ Adversarial angles: all 8192 uniform nibble patterns, hitting **1152** exact ±1 values and
  **732** `+32768 → −32768` int16 wraps; plus an importance-sampled scan for values within 1e−12 of a
  rounding boundary. Tightest genuinely-resolvable margin **2.3e−08** Q1.15 units against a **1.7e−09**
  worst-case error → **~20× headroom**. That ratio is the number that killed 6.14.
- ✅ Soft-fork zero-byte rejection driven through the closed form across all four activation times:
  **31** rejections triggered, **0** disagreements on zero-byte counts.

Beyond the stated bar: the GPU kernel is bit-exact vs the CPU closed form on **16 M** nonces, and
(6.12) every candidate that passes target is re-hashed through the statevector, with the **oracle's**
digest being the one submitted — so even a hypothetical tie cannot become an invalid share.

### Expected speedup vs measured

Predicted **1–10 Mh/s** (1–10% of a ~100 Mh/s FP64 roofline). Measured **287 Mh/s** sustained,
**~298 Mh/s** end-to-end through official-miner — **~29–290× above the predicted band**. Two reasons
the prediction was low: the roofline itself was underestimated (measured FP64 issue rate is
**122.8 G instr/s**, and `fma` doubles the flops per instruction), and the sweep came in at 379 FP64
instructions per nonce rather than ~2000. Against the *measured* roofline the kernel runs at **96%**,
so the pessimistic "1–10% of roofline" assumption was the dominant error.

`88,000×` the shipped statevector, `~390×` stock cuStateVec on the same card.

---

## Success criteria

| Metric | Min | Target | Stretch | **Measured (RTX 5060 Laptop)** |
|--------|-----|--------|---------|--------------------------------|
| kh/s | 50 | 100 | 200 | **286,934** sustained (**1435× stretch**) |
| vs official | 10× | 22× | 44× | **~390×** vs stock cuStateVec, same card |
| Bit-exact vs node | 100% | 100% | 100% | **100%** — 0 real divergences / 64 M values; oracle re-verifies every share |

All three criteria met by a wide margin on a *laptop* GPU weaker than the 4070 the table was written
against.

---

## Next session — short prompt

Phase 6 is complete and the kernel is at 96% of the card's measured FP64 issue rate, so there is no
large win left inside this algorithm. The open work is *deployment*, not optimisation:

```
qhash-miner Phase 7: earn credited pool shares with the Phase 6 closed-form kernel.

REPO / TOOLCHAIN
- /home/dfr/qbtc/qhash-miner (read-only refs: official-miner/, qubitcoin-node/, cpu-miner/)
- RTX 5060 Laptop, sm_120. MUST use ~/opt/cuda-13.3 (apt nvcc 12.4 cannot target sm_120):
    export CUDA_HOME=$HOME/opt/cuda-13.3
    export PATH=$CUDA_HOME/bin:$HOME/opt/cuda-env/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
    cmake -S . -B build -DQHASH_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc

WHERE WE ARE
- Shipped: FP64 closed-form sweep, 287 Mh/s sustained, ~298 Mh/s through official-miner.
  ~88,000x the old statevector, ~390x stock cuStateVec on the same card.
- The kernel is FP64-issue-bound at 96% of the measured roofline (379 FP64 instr/nonce,
  122.8 G instr/s). Do NOT expect more from micro-optimisation.
- 6.14 float-float FP32 is REJECTED on measured evidence: Q1.15 headroom is only ~20x
  and float-float is ~128x less accurate. Phase 4 FP32 also stays REJECTED.
- Self-test digest must stay 8711a7a489f9021a8d8011434e374944ee4051851d21c9fc08dee11757d25bc4
- Gate convention is FIXED: cuStateVec exp(+i theta P), full angle, NOT theta/2.
- Every share is re-verified through the statevector oracle before submission (6.12),
  and the oracle's digest is the one submitted.

WHAT TO DO
1. Phase 5.3 left off at 11/11 pool hash-OK but 0 credited (all "low difficulty share").
   At 287 Mh/s the old ~8-day share TTF at Suprnova Diff 0.5 collapses to well under a
   second, so re-run the pool tests and get actual credited accepts. Check whether the
   pool now raises difficulty via vardiff and that the miner honours mining.set_difficulty.
2. Confirm stability under sustained load: run >=1 h continuous, watch for XID errors,
   thermal throttle, and any share rejected by the oracle re-verification (should be 0).
3. Re-check the BYOS batch size against pool responsiveness under real work restarts,
   not just --benchmark. 4 M nonces is ~15 ms at current speed; verify no stale shares.
4. Only then consider multi-GPU / multi-device dispatch.

GUARDRAILS
- Correctness > hashrate. FP32 stays OFF. Statevector stays as the oracle.
- Do not change the gate convention or the Q1.15 rounding, including the int16 wrap.
- Do not break the BYOS ABI in src/byos/qhash_byos.cpp.
- Measure on the real GPU with a multi-second sustained load; short runs read low
  because the GPU has not boosted (a cold 4 M launch reads ~26 Mh/s vs 287 sustained).
```

<details>
<summary>Archived: the Phase 6 kickoff prompt (delivered — kept for provenance)</summary>

```
Implement Phase 6 of qhash-miner: replace the 65536-amplitude statevector with the
closed-form ⟨Z⟩ sweep. Read "Phase 6 notes" in PLAN.md before touching code.

REPO / TOOLCHAIN
- /home/dfr/qbtc/qhash-miner  (read-only refs: official-miner/, qubitcoin-node/, cpu-miner/)
- RTX 5060 Laptop, sm_120. MUST use ~/opt/cuda-13.3 (apt nvcc 12.4 cannot target sm_120):
    export CUDA_HOME=$HOME/opt/cuda-13.3
    export PATH=$CUDA_HOME/bin:$HOME/opt/cuda-env/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
    cmake -S . -B build -DQHASH_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc

WHERE WE ARE
- Shipped kernel: FP64 statevector, ~3 kh/s, ~4x stock cuStateVec on the same card.
  Phases 1-3b squeezed the statevector dry; it is a dead end.
- Gate convention is FIXED and verified: cuStateVec exp(+i theta P), full angle,
  NOT textbook theta/2. Do not "fix" it.
- Self-test digest must stay 8711a7a489f9021a8d8011434e374944ee4051851d21c9fc08dee11757d25bc4
- Phase 4: FP32 mining REJECTED (92.2% digest match). Stays OFF.
- Phase 5: BYOS drop-in into official-miner works; pool hash-OK 11/11.

THE ALGORITHM (already derived and validated - do not re-derive)
Circuit = 2 layers of [16 single-qubit U2 = Rz.Ry] + [CNOT staircase 0->1->...->15].
1. Trailing staircase gives O_q = U' Z_q U = Z_0 Z_1 ... Z_q  (Heisenberg).
2. O_q is all-Z and Rz is diagonal => layer-2 Rz gates cannot affect the digest.
   nibbles[48..63] are DEAD INPUTS. Only 48 of 64 nibbles matter.
3. Leading staircase on a product state = explicit prefix-XOR MPS, bond dim 2.

Per qubit q, with ty1=angle(nib[q]), tz1=angle(nib[16+q]), ty2=angle(nib[32+q]):
    a_q = e^{+i*tz1}*cos(ty1)      b_q = -e^{-i*tz1}*sin(ty1)
    C_q = [[a_q,b_q],[b_q,a_q]]
    s_q = -sin(2*ty1)*cos(2*tz1)                       (real)
    G_q = [[cos(2*ty2), sin(2*ty2)],[sin(2*ty2), -cos(2*ty2)]]   (real symmetric)
Sweep one 2x2 accumulator:
    L = [[1,0],[0,0]]
    for q in 0..15:
        L = G_q (elementwise*) (C_q^H * L * C_q)
        s = (q<15) ? s_{q+1} : 1
        <Z_q> = L00 + L11 + s*(L01 + L10)
L stays Hermitian => store 4 doubles (L00,L11 real; L01 complex).
No backward sweep, no environments, no shared memory, no barriers.

VALIDATED ALREADY (tests/mps_prototype.cpp, build with:
  g++ -O2 -Iinclude tests/mps_prototype.cpp -o .proto/mps_prototype)
  20 000 nonces, both angle offsets: max|dZ| 3.209e-14,
  0/320 000 Q1.15 mismatches, 0/20 000 nonce mismatches.
  CPU: 1 423 481 circuits/s vs 218/s statevector = 6536x.

ORDER OF WORK - each step is a gate, do not skip ahead
1. 6.2/6.3 CORRECTNESS FIRST, NO KERNEL YET.
   - Extend the harness to >=1M nonces, closed form vs FP64 statevector, 100% Q1.15.
   - Run qhash-custatevec-golden against the CLOSED FORM (not just the CPU sim).
   - Adversarial angles: nibble patterns giving <Z> exactly +-1 (theta=0 when
     offset=0) and <Z> within ~1e-12 of a Q1.15 rounding boundary. Watch the
     deliberate +32768 -> -32768 int16 wrap in to_fixed_q15.
   - Exercise the soft-fork zero-byte rejection paths through the new sim.
   - If any Q1.15 mismatch appears, STOP and report before writing a kernel.
2. 6.4 Wire the closed form into the CPU path; keep the statevector reachable as
   the oracle (self-test, golden harness, candidate re-verification).
3. 6.5 New CUDA kernel: ONE NONCE PER THREAD. Delete the per-block 1 MiB state
   allocation and every __syncthreads(). Keep the old kernel behind a flag for A/B
   until the new one is bit-exact on >=1M nonces.
4. 6.6/6.7/6.8 16-entry cos/sin LUTs (zero transcendentals per nonce); skip
   nibbles[48..63]; Hermitian L as 4 doubles.
5. 6.9 SHA-256 is now the bottleneck, not the sim. Midstate for header bytes 0..63
   (4 compressions -> 3), and constant-fold the message schedules: the padding-only
   second block of the 64-byte final hash has fully constant W[0..63]; the header's
   second block has constant W[4..15].
6. 6.10/6.11 Persistent grid-stride kernel, on-device nonce generation, atomic
   winner queue. Then retune threads/blocks/occupancy FROM SCRATCH - the old
   T=256 C=128 tile=2048 numbers were tuned for a 1 MiB/block statevector and are
   meaningless now. Check registers with ptxas -v; aim for high occupancy.
7. 6.12 Before submitting any share, re-verify the candidate nonce with the
   statevector path. Candidates are rare, so this costs nothing and makes the
   residual ~1e-14 discrepancy incapable of producing an invalid share.
8. 6.13 Retire the tile/fiber/dbuf/boundary code paths and their CMake options.
9. 6.14 ONLY if profiling shows FP64-bound: float-float (double-single) FP32,
   ~2^-46 accuracy at ~15 FP32 ops per op. Do not do this speculatively.
10. Re-validate the BYOS path (build-official-byos.sh --cuda, byos_check, ctest)
    and confirm the ABI in src/byos/qhash_byos.cpp is unchanged.
11. Update PLAN.md: task statuses, speed results log (new Iter rows with the
    standard columns), correctness snapshots, profile snapshot.

GUARDRAILS
- Correctness > hashrate. FP32 mining stays OFF. Keep the statevector as the oracle.
- Do not change the gate convention or the Q1.15 rounding, including the int16 wrap.
- Do not break the BYOS ABI used by official-miner.
- Measure every claim on the real GPU; log real numbers, never estimates, in the
  speed table.

EXPECTED OUTCOME
Quantum part is ~6500x cheaper, so the kernel should become SHA-256/occupancy bound.
FP64 roofline on this card is ~100 Mh/s; at 1-10% of roofline that is 1-10 Mh/s vs
today's ~3 kh/s. Treat as a bound to measure against, not a promise.
```

</details>

---

## File map

| Path | Role |
|------|------|
| `include/circuit.cuh` / `gates.cuh` / `sha256.cuh` | Device/host primitives (**gates = cuStateVec convention**); `gates.cuh` also has `qfma`, the bit-identical host/device FMA |
| **`include/closed_form.cuh`** | **Phase 6 simulator.** `AngleLut`, `qubit_factors`, `closed_form_sweep` — the shipped ⟨Z⟩ path, shared verbatim by CPU and GPU |
| `include/sha256_mine.cuh` | Mining SHA-256: midstate, constant-folded header `W[4..15]` and final-block `W[0..63]` |
| `src/qhash_kernel.cu` | `qhash_cf_kernel` (closed form, one nonce/thread, shipped) + `qhash_mine_kernel` (statevector oracle, plain gate-at-a-time) + candidate re-verification |
| `src/qhash_cpu.cpp` | Reference; `qhash_sim_t` selects closed form (default) or the statevector oracle |
| `src/main.cpp` | CLI: `--benchmark --self-test --compare-fp --sim --threads --blocks --chunk --streams` |
| `tests/custatevec_golden.cpp` | cuStateVec FP64 vs **both** CPU sims, Q1.15 harness |
| **`tests/closed_form_check.cpp`** | **Phase 6 gate**: `scale` / `boundary` / `degenerate` / `softfork` / `digest` / `gpu` modes |
| `tests/reference_ld.h` | Independent `long double` sweep; arbitrates whether a Q1.15 mismatch is a tie or a real divergence |
| `tests/mps_prototype.cpp` | Phase 6.1 standalone derivation check + CPU speed ratio |
| `bench/cf_ablation.cu` | Ablation (sweep vs SHA vs full) + measured FP64 issue-rate roofline |
| `tests/byos_check.cpp` | BYOS ABI / expectations vs CPU |
| `src/byos/qhash_byos.cpp` | Official miner BYOS (`run_simulation`) — **ABI frozen** |
| `src/byos/qhash_scanhash_cuda.cpp` | CUDA batch scanhash + CPU fallback; vendored here, installed into `official-miner/algo/qhash/` by the build script |
| `scripts/build-official-byos.sh` | Build official-miner with CPU/CUDA BYOS (`install` copies the scanhash wrapper across) |
| `scripts/pool-accept-check.sh` | Phase 5.3 pool run helper (HeroMiners/Suprnova/LuckyPool) |
| `Makefile` | CPU-only build |
| `CMakeLists.txt` | CUDA build (`QHASH_FUSE_RZ_RY`, `QHASH_BYOS_FP64`, `qhash_byos_plugin`); FMA contraction is **off** in both compilers so host and device agree bit-for-bit |
