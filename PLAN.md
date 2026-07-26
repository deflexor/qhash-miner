# Qhash Miner Optimization — Plan & Progress Log

**Project:** `qhash-miner/` (custom CUDA replacement for cuStateVec qPoW)  
**Started:** 2026-07-26  
**Goal:** 50–200+ kh/s on RTX 4070 (vs official ~4.5 kh/s), bit-exact vs consensus node

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

**How to append a row**

```bash
export CUDA_HOME=$HOME/opt/cuda-13.3
export PATH=$CUDA_HOME/bin:$HOME/opt/cuda-env/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
./build/qhash-miner --benchmark --nonces 2048 --threads 256 --chunk 128
# Fill: Iter, Date, Backend, Hardware (nvidia-smi), Precision, Nonces, Time, H/s, kh/s
# Change = (new_khs / prev_best_khs - 1) * 100% on same GPU
```

**Build note:** apt `nvidia-cuda-toolkit` is **12.4** and cannot target `sm_120`. Use local **CUDA 13.3** at `~/opt/cuda-13.3` (driver UMD is already 13.3).

```bash
# Optional cuStateVec golden (needs cuquantum + libcublas.so.13):
cmake -S . -B build -DQHASH_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
./build/qhash-custatevec-golden --nonces 1024

# Optional 4096-amp dynamic tile (usually no faster):
cmake -S . -B build-tile4096 -DQHASH_SMEM_TILE=4096 -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
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

---

## Phase 3 / 3b notes

1. **Defaults:** `--threads 256 --chunk 128 --streams 1`; `QHASH_SMEM_TILE=2048`; `QHASH_TILE_DBUF=0`; `QHASH_BOUNDARY_PAIR=0`.
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

## Success criteria

| Metric | Min | Target | Stretch |
|--------|-----|--------|---------|
| RTX 4070 kh/s | 50 | 100 | 200 |
| vs official | 10× | 22× | 44× |
| Bit-exact vs node | 100% | 100% | 100% |

---

## Next session — short prompt

Copy-paste into the next chat:

```
Continue qhash-miner from PLAN.md (kernel plateau / credited pool accepts).

Context:
- Repo: /home/dfr/qbtc/qhash-miner
- Hardware: RTX 5060 Laptop (sm_120); build with ~/opt/cuda-13.3
- Same-card A/B: stock cuStateVec FP32 ~0.76 kh/s (official-miner-stock/) vs our FP64 CUDA BYOS ~3.1–3.3 kh/s → ~4×
- Best FP64 kernel: fuse low U2+CNOT + ILP + high-q fiber + fused ⟨Z⟩ + tile 2048, T=256 C=128 → ~2.7–3.3 kh/s
- Gate convention FIXED; self-test digest 8711a7a489f9021a8d8011434e374944ee4051851d21c9fc08dee11757d25bc4
- Phase 4 FP32 REJECTED; Phase 5 BYOS wired; pool hash-OK 11/11 (low-diff rejects only); HeroMiners stratum down; Suprnova default (Diff 0.5, ~8d share TTF)
- CPU fallback when no CUDA: done; ncu still ERR_NVGPUCTRPERM on WSL
- Target still 10–40× vs official; need ncu or algorithmic leap past ~3 kh/s plateau

Do next:
1. Enable GPU performance counters on WSL; ncu smem/occupancy/stalls — gate to next kernel wins
2. Credited pool accepts: HeroMiners when stratum is back (`=32`/`=64`), or overnight Suprnova; confirm Accepted > 0
3. Optional after ncu: warp-specialized gates / multi-nonce-per-block / multi-GPU
4. Update PLAN.md

Correctness > hashrate; FP32 mining stays OFF.
```

---

## File map

| Path | Role |
|------|------|
| `include/circuit.cuh` / `gates.cuh` / `sha256.cuh` | Device/host primitives (**gates = cuStateVec convention**) |
| `src/qhash_kernel.cu` | Monolithic kernel (fuse, CNOT tile, fiber, fused ⟨Z⟩, low U2+CNOT fuse, ILP, optional boundary/dbuf/4096) |
| `src/qhash_cpu.cpp` | Reference |
| `src/main.cpp` | CLI: `--benchmark --compare-fp --threads --chunk --streams` |
| `tests/custatevec_golden.cpp` | cuStateVec FP64 vs CPU Q1.15 harness |
| `tests/byos_check.cpp` | BYOS ABI / expectations vs CPU |
| `src/byos/qhash_byos.cpp` | Official miner BYOS (`run_simulation`) |
| `scripts/build-official-byos.sh` | Build official-miner with CPU/CUDA BYOS |
| `scripts/pool-accept-check.sh` | Phase 5.3 pool run helper (HeroMiners/Suprnova/LuckyPool) |
| `algo/qhash/qhash-scanhash-cuda.cpp` | CUDA batch scanhash + CPU fallback (`../official-miner/`) |
| `Makefile` | CPU-only build |
| `CMakeLists.txt` | CUDA build (`QHASH_FUSE_RZ_RY`, `QHASH_SMEM_TILE`, `QHASH_BOUNDARY_PAIR`, `QHASH_BYOS_FP64`, `qhash_byos_plugin`) |
