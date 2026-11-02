#!/bin/bash
# cf_build: three builds of the night-4 exit, each with the diagnostic (diag.py), CPU only:
#   cf-t234  production's branch (kvshare-up) + upstream #234 alone
#   cf-t3    production's branch + #226 #246 #217 (kvshare-up2 without #234)
#   cf-up2   kvshare-up2 as tested in night 4
# Binaries in build/apps-cf-*; the tree goes back to kvshare-up2 at the end.
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd /opt/NInferno-up
G() { git -c user.name="Leon Bollerup" -c user.email=mail@r2x2.com "$@"; }
step() { echo "=== $(date +%T) $*"; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tree not clean"; exit 1; }
mk() {  # name base [commits to pick...]
    local n=$1 base=$2; shift 2
    git checkout -q -B $n $base || return 1
    for c in "$@"; do G cherry-pick -x $c > /dev/null || { git cherry-pick --abort; echo "PICK_FAILED $c"; return 1; }; done
    python3 /root/gbench/cf/diag.py /opt/NInferno-up || return 1
    G commit -qam "diag: a resource subtraction underflow names its call site and operands"
    step "build $n: $(git log --oneline -1 HEAD~1 | cut -c1-70)"
    if nice -n 15 ninja -C build -j6 ninferno-serve > /root/gbench/cf/build_$n.log 2>&1; then
        mkdir -p build/apps-$n && cp build/apps/ninferno-serve build/apps-$n/ && echo "BUILD_OK $n"
    else echo "BUILD_FAILED $n"; grep -m8 error /root/gbench/cf/build_$n.log; return 1; fi
}
mk cf-t234 kvshare-up 6c0e52ad
mk cf-t3   kvshare-up 3bfa4acf 19775e33 2dcd1515
mk cf-up2  kvshare-up2
git checkout -q kvshare-up2
step "CF_BUILD_DONE"
