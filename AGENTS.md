# Agent notes — qhash-miner

## CUDA toolchain (use this, not apt 12.4)

| Item | Location / version |
|------|--------------------|
| **CUDA_HOME** | `/home/dfr/opt/cuda-13.3` (symlinks into the uv venv) |
| **nvcc** | `$CUDA_HOME/bin/nvcc` — **13.3.73** |
| **headers** | `$CUDA_HOME/include` |
| **libs** | `$CUDA_HOME/lib` (`libcudart.so.13`, etc.) |
| Real package root | `/home/dfr/opt/cuda-env/lib/python3.12/site-packages/nvidia/cu13/` |
| uv venv | `/home/dfr/opt/cuda-env` |

```bash
export CUDA_HOME=$HOME/opt/cuda-13.3
export PATH=$CUDA_HOME/bin:$HOME/opt/cuda-env/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
cmake -S . -B build -DQHASH_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
```

**Do not use** `/usr/bin/nvcc` (apt **12.4.131**) — cannot target `sm_120` (RTX 50xx).

## Versions in use

| Component | Version |
|-----------|---------|
| Driver / CUDA UMD | 610.62 / **13.3** |
| GPU | RTX 5060 Laptop — compute **12.0** (`sm_120`) |
| nvidia-cuda-nvcc | 13.3.73 |
| nvidia-cuda-runtime | 13.3.29 |
| nvidia-cuda-cccl | 13.3.3.4.1 |
| nvidia-nvjitlink | 13.3.33 |
| nvidia-cuda-cupti | 13.3.75 |
| cmake (venv) | 4.4.0 |
| g++ (host) | 15.2.0 |
| Nsight Compute (`ncu`) | system apt 2024.1 (with CUDA 12.4 toolkit) — needs `ERR_NVGPUCTRPERM` fix on WSL |
| custatevec-cu13 (optional golden) | 1.14.0 at `…/cuquantum/` — needs **cuBLAS 13** to link |

## Gate convention (critical)

`custatevecApplyPauliRotation(θ, P)` in cuStateVec **1.14** (node + miner) behaves as **`exp(+i θ P)`** with full angle in `cos/sin`, **not** the textbook/`exp(-i θ/2 P)` form. `include/gates.cuh` matches the library. Validate with `./build/qhash-custatevec-golden --nonces 1024`.

## Project libs

No third-party C++ deps beyond **CUDA Runtime** (`CUDA::cudart`) and **libm**. Optional golden adds **cuStateVec** + **cuBLAS**. Algorithm/crypto are in-tree (`include/`, `src/`).

See `PLAN.md` for phase status and the next-session prompt.

## Extra

if you need python (i hope not) then use `uv` tool and local venv (~/opt/cuda-env/)
