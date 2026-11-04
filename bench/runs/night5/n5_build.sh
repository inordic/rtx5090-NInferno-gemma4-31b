#!/bin/bash
# n5_build: patch 19 on kvshare-up4 -> build/apps-up4, the CPU unit tests, and a counting build -> build/apps-sk.
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd /opt/NInferno-up
G() { git -c user.name="Leon Bollerup" -c user.email=mail@r2x2.com "$@"; }
step() { echo "=== $(date +%T) $*"; }
if ! git diff --quiet; then
    G commit -qam "gemma4: a mid-prompt prefill call stops the last layer at its cache write

The last layer is a full-attention layer no later layer attends through, and on the staged prompt
path only the prompt's last call and a call ending on a capture's frontier have their hidden read.
Every other call's last-layer attention, O projection, MLP and final norm were thrown away. A
rebuild reads every call's tail and a causal score every column, so they keep the full pass.
Opt-in: NINFERNO_GEMMA4_SKIP_LAST_LAYER=1."
fi
step "build $(git log --oneline -1 | cut -c1-90)"
if nice -n 15 ninja -C build -j6 ninferno-serve ninferno-gemma4 ninferno ninferno-perplexity ninferno-convert > /root/gbench/n5/build_up4.log 2>&1; then
    mkdir -p build/apps-up4 && cp build/apps/ninferno-serve build/apps/ninferno-gemma4 build/apps/ninferno build/apps-up4/ && echo BUILD_up4_OK
else echo BUILD_up4_FAILED; grep -m8 -E "error|FAILED" /root/gbench/n5/build_up4.log | cut -c1-200; step N5_BUILD_DONE; exit 1; fi
step "CPU unit tests"
cd build
T="tests/ninferno_openai_schema_test tests/ninferno_openai_responses_test tests/ninferno_serve_options_test
   tests/ninferno_resource_manager_test tests/ninferno_context_cache_defaults_test
   tests/ninferno_qwen3_5_frontend_test tests/ninferno_qwen3_5_state_image_layout_test"
G2="tests/ninferno_qwen3_5_state_image_test tests/ninferno_qwen3_5_context_store_test"
if nice -n 15 ninja -j6 $T $G2 > /root/gbench/n5/build_up4_tests.log 2>&1; then
    for t in $T; do ./$t > /root/gbench/n5/tu4_$(basename $t).log 2>&1 && echo "PASS $t" || { echo "FAIL $t"; tail -3 /root/gbench/n5/tu4_$(basename $t).log | cut -c1-200; }; done
else echo TESTS_BUILD_FAILED; grep -m5 -E "error|FAILED" /root/gbench/n5/build_up4_tests.log | cut -c1-200; fi
cd ..
step "counting build"
python3 /root/gbench/n5/n5_skipcount.py /opt/NInferno-up || exit 1
if nice -n 15 ninja -C build -j6 ninferno-serve > /root/gbench/n5/build_sk.log 2>&1; then
    mkdir -p build/apps-sk && cp build/apps/ninferno-serve build/apps-sk/ && echo BUILD_sk_OK
else echo BUILD_sk_FAILED; grep -m8 -E "error|FAILED" /root/gbench/n5/build_sk.log | cut -c1-220; fi
git checkout -- src/ops/softmax_attention/gemma4/kernel.cuh src/ops/softmax_attention/gemma4/launch.cu
nice -n 15 ninja -C build -j6 ninferno-serve > /root/gbench/n5/build_restore.log 2>&1 && echo TREE_RESTORED
df -h / | tail -1
step N5_BUILD_DONE
