#!/bin/bash
# the same tree with CUDA 13.4: W8 stopped on a missing NVTX header
cd /opt/NInferno-up
export PATH=/usr/local/cuda-13.4/bin:$PATH
cmake -S . -B build-134 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc -DBUILD_TESTING=OFF -DNINFERNO_BUILD_BENCHMARKS=OFF 2>&1 | tail -1 | cut -c1-160
if nice -n 15 ninja -C build-134 -j7 ninferno-serve > /root/gbench/n5/build_134.log 2>&1; then
    mkdir -p build/apps-134 && cp build-134/apps/ninferno-serve build/apps-134/ && echo BUILD_134_OK
else echo BUILD_134_FAILED; grep -m4 -E "error|FAILED" /root/gbench/n5/build_134.log | cut -c1-200; fi
rm -rf build-134
echo F_BUILD_DONE
