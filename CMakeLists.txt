cmake_minimum_required(VERSION 3.18)
project(qhash-miner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QHASH_ENABLE_CUDA "Build CUDA mining kernel" ON)
option(QHASH_FUSE_RZ_RY "Fuse Rz·Ry into one 2×2 per qubit/layer (Phase 2)" ON)
option(QHASH_ENABLE_CUSTATEVEC "Build cuStateVec FP64 golden harness" ON)
set(QHASH_SMEM_TILE "2048" CACHE STRING "Shared-memory tile amps (0|1024|2048|4096); 4096 needs dynamic smem for FP64")
option(QHASH_BOUNDARY_PAIR "Dual-tile low section so CNOT(tileQ-1,tileQ) stays in smem (usually slower)" OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(QHASH_COMMON_SRC
    src/qhash_cpu.cpp
)

# ---- CUDA detection ----
set(QHASH_HAS_CUDA OFF)
if(QHASH_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(QHASH_HAS_CUDA ON)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS "CUDA not found — building CPU-only (QHASH_CPU_ONLY)")
    endif()
endif()

if(QHASH_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(qhash_core STATIC
        ${QHASH_COMMON_SRC}
        src/qhash_kernel.cu
    )
    set_target_properties(qhash_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        # 75=Turing, 86=Ampere, 89=Ada, 90=Hopper, 120=Blackwell (RTX 50xx)
        CUDA_ARCHITECTURES "75;86;89;90;120"
    )
    target_compile_options(qhash_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_link_libraries(qhash_core PUBLIC CUDA::cudart)
else()
    # Compile the .cu file as C++ with CPU fallback path.
    add_library(qhash_core STATIC
        ${QHASH_COMMON_SRC}
        src/qhash_kernel.cu
    )
    set_source_files_properties(src/qhash_kernel.cu PROPERTIES LANGUAGE CXX)
    target_compile_definitions(qhash_core PUBLIC QHASH_CPU_ONLY)
endif()

if(QHASH_FUSE_RZ_RY)
    target_compile_definitions(qhash_core PUBLIC QHASH_FUSE_RZ_RY)
endif()

target_compile_definitions(qhash_core PUBLIC QHASH_SMEM_TILE=${QHASH_SMEM_TILE})
if(QHASH_BOUNDARY_PAIR)
    target_compile_definitions(qhash_core PUBLIC QHASH_BOUNDARY_PAIR=1)
else()
    target_compile_definitions(qhash_core PUBLIC QHASH_BOUNDARY_PAIR=0)
endif()
message(STATUS "QHASH_SMEM_TILE=${QHASH_SMEM_TILE} QHASH_BOUNDARY_PAIR=${QHASH_BOUNDARY_PAIR}")

target_include_directories(qhash_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qhash_core PUBLIC m)

add_executable(qhash-miner src/main.cpp)
target_link_libraries(qhash-miner PRIVATE qhash_core)

add_executable(qhash-test tests/test_circuit.cpp)
target_link_libraries(qhash-test PRIVATE qhash_core)

add_executable(qhash-bench bench/benchmark.cpp)
target_link_libraries(qhash-bench PRIVATE qhash_core)

add_library(qhash_byos STATIC src/byos/qhash_byos.cpp)
target_link_libraries(qhash_byos PUBLIC qhash_core)
option(QHASH_BYOS_FP64 "BYOS run_simulation uses FP64 + header nTime angles" OFF)
if(QHASH_BYOS_FP64)
    target_compile_definitions(qhash_byos PUBLIC QHASH_BYOS_FP64=1)
endif()

add_executable(qhash-byos-check tests/byos_check.cpp)
target_link_libraries(qhash-byos-check PRIVATE qhash_byos)
add_test(NAME byos_check COMMAND qhash-byos-check)

# Shared plugin for official-miner CUDA BYOS (device link baked into .so).
if(QHASH_HAS_CUDA)
    add_library(qhash_byos_plugin SHARED
        src/qhash_cpu.cpp
        src/qhash_kernel.cu
        src/byos/qhash_byos.cpp
    )
    set_target_properties(qhash_byos_plugin PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;86;89;90;120"
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME qhash_byos_plugin
    )
    target_compile_options(qhash_byos_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_include_directories(qhash_byos_plugin PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(qhash_byos_plugin PUBLIC CUDA::cudart m)
    target_compile_definitions(qhash_byos_plugin PUBLIC QHASH_SMEM_TILE=${QHASH_SMEM_TILE})
    if(QHASH_FUSE_RZ_RY)
        target_compile_definitions(qhash_byos_plugin PUBLIC QHASH_FUSE_RZ_RY)
    endif()
    if(QHASH_BOUNDARY_PAIR)
        target_compile_definitions(qhash_byos_plugin PUBLIC QHASH_BOUNDARY_PAIR=1)
    else()
        target_compile_definitions(qhash_byos_plugin PUBLIC QHASH_BOUNDARY_PAIR=0)
    endif()
    if(QHASH_BYOS_FP64)
        target_compile_definitions(qhash_byos_plugin PUBLIC QHASH_BYOS_FP64=1)
    endif()
endif()

# ---- Optional cuStateVec golden (node-style FP64) ----
set(QHASH_HAS_CUSTATEVEC OFF)
if(QHASH_ENABLE_CUSTATEVEC AND QHASH_HAS_CUDA)
    set(_CUQ_ROOT "$ENV{HOME}/opt/cuda-env/lib/python3.12/site-packages/cuquantum")
    if(DEFINED ENV{CUQUANTUM_ROOT} AND EXISTS "$ENV{CUQUANTUM_ROOT}/include/custatevec.h")
        set(_CUQ_ROOT "$ENV{CUQUANTUM_ROOT}")
    endif()
    find_path(CUSTATEVEC_INCLUDE_DIR custatevec.h
        HINTS "${_CUQ_ROOT}/include" /usr/include /usr/local/include)
    find_library(CUSTATEVEC_LIBRARY
        NAMES custatevec custatevec.so.1 libcustatevec.so.1
        HINTS "${_CUQ_ROOT}/lib" /usr/lib /usr/local/lib)
    if(CUSTATEVEC_INCLUDE_DIR AND CUSTATEVEC_LIBRARY AND TARGET CUDA::cublas)
        set(QHASH_HAS_CUSTATEVEC ON)
        add_executable(qhash-custatevec-golden tests/custatevec_golden.cpp)
        target_include_directories(qhash-custatevec-golden PRIVATE ${CUSTATEVEC_INCLUDE_DIR})
        target_link_libraries(qhash-custatevec-golden PRIVATE qhash_core ${CUSTATEVEC_LIBRARY} CUDA::cublas CUDA::cudart)
        message(STATUS "cuStateVec golden: ${CUSTATEVEC_LIBRARY}")
    else()
        message(STATUS "cuStateVec not found — skipping qhash-custatevec-golden")
    endif()
endif()

enable_testing()
add_test(NAME circuit COMMAND qhash-test)
add_test(NAME selftest COMMAND qhash-miner --self-test)
if(QHASH_HAS_CUSTATEVEC)
    add_test(NAME custatevec_golden COMMAND qhash-custatevec-golden --nonces 64)
endif()

install(TARGETS qhash-miner RUNTIME DESTINATION bin)
