#!/bin/bash
# CUDA builds of llama.cpp and ik_llama.cpp for sm_120 (low priority, background)
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd /opt
echo "=== llama.cpp $(date +%T)"
cd /opt/llama.cpp && git pull -q && nice -n 19 cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FA_ALL_QUANTS=ON >/dev/null && nice -n 19 cmake --build build-cuda -j4 --target llama-server llama-bench llama-cli llama-perplexity 2>&1 | grep -E "error|Built target" ; ls -la build-cuda/bin/ | grep -E "server|bench"
echo "=== ik_llama.cpp $(date +%T)"
cd /opt && (test -d ik_llama.cpp || git clone -q --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git) && cd ik_llama.cpp && git log --oneline -1 && nice -n 19 cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FA_ALL_QUANTS=ON >/dev/null && nice -n 19 cmake --build build -j4 --target llama-server llama-bench llama-perplexity 2>&1 | grep -E "error|Built target"; ls -la build/bin/ | grep -E "server|bench"
echo "=== BUILDS DONE $(date +%T)"
