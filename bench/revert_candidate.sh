#!/bin/bash
# revert_candidate.sh: put back the newest gemma4-31b-fast.service backup and restart it.
set -e
U=/etc/systemd/system/gemma4-31b-fast.service
B=$(ls -t $U.bak-* | head -1)
cp -p $B $U
U2=/etc/systemd/system/gemma4-31b-262k.service
B2=$(ls -t $U2.bak-* 2>/dev/null | head -1)
[ -n "$B2" ] && cp -p $B2 $U2
systemctl daemon-reload
/root/gbench/deploy_fast.sh
grep ExecStart $U | cut -c1-200
