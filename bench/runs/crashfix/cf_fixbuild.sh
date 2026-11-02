#!/bin/bash
# cf_fixbuild: kvshare-up3 = kvshare-up2 + the capture fix (fix.py) + the underflow diagnostic
# (diag.py) -> build/apps-up3, then the CPU unit tests and the GPU tests' builds.
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd /opt/NInferno-up
G() { git -c user.name="Leon Bollerup" -c user.email=mail@r2x2.com "$@"; }
step() { echo "=== $(date +%T) $*"; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tree not clean"; exit 1; }
git checkout -q -B kvshare-up3 kvshare-up2
python3 /root/gbench/cf/fix.py /opt/NInferno-up || exit 1
G commit -qam "program: a turn-end capture publishes privately when a shared copy would exceed the active entitlement

Since upstream #234 (DECISIONS 63) the shared candidate after the last message and the turn
closure's private capture form one group at the closed turn's end. A conversation that continues
from its own checkpoint keeps that prefix outside its active entitlement (request_plan), so pages of
it the active sequence alone still references counted as its unique pages: the shared copy moved
more pages out of the entitlement than it held, and committing the capture stopped the engine
(resource subtraction underflow, capture.cpp). Such a group now publishes only its private
checkpoint, as the turn closure did before."
python3 /root/gbench/cf/diag.py /opt/NInferno-up || exit 1
G commit -qam "program: a resource subtraction underflow names its call site and operands"
step "build $(git log --oneline -3 | tr '\n' ' ' | cut -c1-160)"
if nice -n 15 ninja -C build -j6 ninferno-serve ninferno-gemma4 ninferno > /root/gbench/cf/build_up3.log 2>&1; then
    mkdir -p build/apps-up3 && cp build/apps/ninferno-serve build/apps/ninferno-gemma4 build/apps/ninferno build/apps-up3/ && echo BUILD_up3_OK
else echo BUILD_up3_FAILED; grep -m8 error /root/gbench/cf/build_up3.log; exit 1; fi
step "CPU unit tests"
cd build
T="tests/ninferno_openai_schema_test tests/ninferno_openai_responses_test tests/ninferno_serve_options_test
   tests/ninferno_resource_manager_test tests/ninferno_context_cache_defaults_test tests/ninferno_decision_fingerprint_test
   tests/ninferno_qwen3_5_frontend_test tests/ninferno_qwen3_5_state_image_layout_test"
G2="tests/ninferno_qwen3_5_state_image_test tests/ninferno_qwen3_5_context_store_test"
if nice -n 15 ninja -j6 $T $G2 > /root/gbench/cf/build_up3_tests.log 2>&1; then
    for t in $T; do ./$t > /root/gbench/cf/tu3_$(basename $t).log 2>&1 && echo "PASS $t" || { echo "FAIL $t"; tail -3 /root/gbench/cf/tu3_$(basename $t).log; }; done
else echo TESTS_BUILD_FAILED; grep -m5 error /root/gbench/cf/build_up3_tests.log; fi
step "UP3_DONE"
