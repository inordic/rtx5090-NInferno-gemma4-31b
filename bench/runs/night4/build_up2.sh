#!/bin/bash
# build_up2.sh: kvshare-up2 (/opt/NInferno-up: kvshare-up + upstream #226 TMA fence, #246 ring live
# window, #217 OOV token ends its request, #234 boundary at the closed turn's end) -> build/apps-up2,
# then the CPU unit tests. The GPU tests (state image, context store, NVFP4 A4 linears) run in the
# night batch, when the GPU is free.
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd /opt/NInferno-up/build
echo "=== $(date +%T) build $(git -C /opt/NInferno-up log --oneline -1)"
if nice -n 15 ninja -j5 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/build_up2.log 2>&1; then
  mkdir -p apps-up2 && cp apps/ninferno-serve apps/ninferno-gemma4 apps/ninferno apps-up2/ && echo BUILD_up2_OK
else echo BUILD_up2_FAILED; grep -m8 -E "error" /root/gbench/build_up2.log; exit 1; fi
echo "=== $(date +%T) CPU unit tests"
T="tests/ninferno_openai_schema_test tests/ninferno_openai_responses_test tests/ninferno_serve_options_test
   tests/ninferno_resource_manager_test tests/ninferno_context_cache_defaults_test tests/ninferno_decision_fingerprint_test
   tests/ninferno_qwen3_5_frontend_test tests/ninferno_qwen3_5_state_image_layout_test"
G="tests/ninferno_qwen3_5_state_image_test tests/ninferno_qwen3_5_context_store_test tests/ninferno_linear_nvfp4_a4_test
   tests/ninferno_linear_add_nvfp4_test tests/ninferno_linear_gemma4_nvfp4_test"
if nice -n 15 ninja -j5 $T $G > /root/gbench/build_up2_tests.log 2>&1; then
  for t in $T; do ./$t > /root/gbench/tu2_$(basename $t).log 2>&1 && echo "PASS $t" || { echo "FAIL $t"; tail -5 /root/gbench/tu2_$(basename $t).log; }; done
else echo TESTS_BUILD_FAILED; grep -m5 error /root/gbench/build_up2_tests.log; fi
echo "=== $(date +%T) UP2_DONE"
