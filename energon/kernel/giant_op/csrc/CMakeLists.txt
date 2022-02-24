cmake_minimum_required(VERSION 3.13)
project(energon_giant_operation LANGUAGES CXX CUDA)

# Find de-libiary
set(CMAKE_PREFIX_PATH "/home/lcdjs/anaconda3/envs/py38/lib/python3.8/site-packages/torch/share/cmake")
list(APPEND CMAKE_PREFIX_PATH "/home/lcdjs/ColossalAI-Inference/energon/kernel/giant_op/csrc")
set(NCCL_DIR "/opt/lcsoftware/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.3.0/nccl-2.9.6-1-ysovaavjkgjez2fwms4dkvatu5yrxbec")

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(NCCL REQUIRED)

# Find Python
set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());" RESULT_VARIABLE _PYTHON_SUCCESS OUTPUT_VARIABLE PY_INCLUDE_DIR)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${NCCL_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PY_INCLUDE_DIR})

set(SOURCES 
    fastertransformer/th_op/decoder.cc
)


add_library(FTdecoder SHARED ${SOURCES})

target_include_directories(FTdecoder
    PRIVATE ${PROJECT_SOURCE_DIR}
)

set(CMAKE_CXX_STANDARD 14)

target_link_libraries(FTdecoder PUBLIC ${NCCL_LIBRARIES} ${MPI_LIBRARIES})


set(USING_WMMA True)

set(USING_WMMA True)
set(CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80;-G;-g)

if(USING_WMMA STREQUAL True)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  message("-- Use WMMA")
endif()


# Remove FT limitation 
add_definitions("-DBUILD_GPT")
add_definitions("-DBUILD_PYT")
add_definitions("-DTORCH_CUDA")
add_definitions("-DUSE_C10D_NCCL")

if (${CUDA_VERSION} GREATER_EQUAL 11.0)
  message(STATUS "Add DCUDA11_MODE")
  add_definitions("-DCUDA11_MODE")
endif()


#ifdef BUILD_GPT