#!/bin/bash
# kvshare-up4 = kvshare-up3 + six upstream fixes + the revert of patch 12 (upstream #318 carries its fix).
cd /opt/NInferno-up
G() { git -c user.name="Leon Bollerup" -c user.email=mail@r2x2.com "$@"; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tree not clean"; exit 1; }
git checkout -q -B kvshare-up4 kvshare-up3
P12=$(git log --format=%h --grep="an image starts at its own" -1 kvshare-up3)
echo "patch 12: $P12 $(git log -1 --format=%s $P12 | cut -c1-80)"
G cherry-pick -x -m 1 281ec982 > /dev/null && echo "ok #324"
G cherry-pick -x      c6e54d71 > /dev/null && echo "ok #326 capture"
G cherry-pick -x -m 1 cfd61ef5 > /dev/null && echo "ok #327"
G cherry-pick -x -m 1 76db073f > /dev/null && echo "ok #325"
G revert --no-edit $P12 > /dev/null && echo "ok revert 12"
if G cherry-pick -x -m 1 7fe1dece > /dev/null 2>/tmp/n5pick.err; then echo "ok #318"; else
    echo "#318 conflict:"; git status --porcelain | grep -E "^(UU|AA|U|.U)"; git diff --name-only --diff-filter=U
    git diff | sed -n 1,40p
    exit 2
fi
G cherry-pick -x -m 1 9d8a8f29 > /dev/null && echo "ok #319"
git log --oneline kvshare-up3..HEAD | cut -c1-90
