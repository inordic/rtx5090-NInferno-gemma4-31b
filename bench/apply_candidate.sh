#!/bin/bash
# apply_candidate.sh: switch gemma4-31b-fast to the overnight candidate - binary apps-final3
# (NInferno 1a3d564: FP8 attention, qkv eight-token schedule, Q6 head 8-column tile) with 7 MTP drafts instead of 8.
# The current unit is kept as gemma4-31b-fast.service.bak-<date>; revert_candidate.sh restores it.
set -e
U=/etc/systemd/system/gemma4-31b-fast.service
cp -p $U $U.bak-$(date +%Y%m%d-%H%M%S)
sed -i -e "s#/opt/NInferno/build/apps-final/ninferno-serve#/opt/NInferno/build/apps-final3/ninferno-serve#" \
       -e "s#--draft-tokens 8#--draft-tokens 7#" $U
# The 262K unit (not started here) gets the same binary and draft count.
U2=/etc/systemd/system/gemma4-31b-262k.service
cp -p $U2 $U2.bak-$(date +%Y%m%d-%H%M%S)
sed -i -e "s#/opt/NInferno/build/apps-final/ninferno-serve#/opt/NInferno/build/apps-final3/ninferno-serve#" \
       -e "s#--draft-tokens 8#--draft-tokens 7#" $U2
systemctl daemon-reload
/root/gbench/deploy_fast.sh
grep ExecStart $U | cut -c1-200
