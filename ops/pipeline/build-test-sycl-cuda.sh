#!/bin/bash
## Build one library with both CUDA and SYCL and run the SYCL C++ tests on the CPU.

set -euox pipefail

mkdir build
pushd build
cmake .. -DGOOGLE_TEST=ON -DUSE_CUDA=ON -DPLUGIN_SYCL=ON -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CUDA_HOST_COMPILER=g++ -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--no-undefined \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -GNinja
ninja
popd

echo "--- Check that both device images are in the library"
sections=$(readelf -S -W lib/libxgboost.so)
grep -q '\.nv_fatbin' <<< "${sections}"
grep -q '\.tgtimg' <<< "${sections}"
readelf -d lib/libxgboost.so | grep NEEDED

# No GPU on the runner. Tests in a CUDA build that need a GPU abort without one, so only the SYCL
# tests (on the CPU device) and the storage backend tests run here.
echo "--- Run C++ tests"
./build/testxgboost --gtest_filter='Sycl*:HostDeviceVectorBackends*'
