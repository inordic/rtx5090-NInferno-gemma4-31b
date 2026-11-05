#!/bin/bash
# deploy_up4.sh: the fast unit on apps-up4 (kvshare-up4: six upstream fixes, patch 12 reverted, patch 19
# on), the assistant in NVFP4, prefill chunk 1280, 203,136 tokens (night 5 report). Keeps the old unit
# as .bak, restarts when no request is in flight, waits until 8080 answers.
set -e
U=/etc/systemd/system/gemma4-31b-fast.service
BAK=$U.bak-$(date +%Y%m%d-%H%M%S)
cp -p $U $BAK
echo "backup: $BAK"
grep -q '^ExecStart=/opt/NInferno-up/build/apps-up3/ninferno-serve ' $U
sed -i -e 's#^ExecStart=/opt/NInferno-up/build/apps-up3/ninferno-serve #ExecStart=/opt/NInferno-up/build/apps-up4/ninferno-serve #' \
       -e 's#--max-context 202112 --prefill-chunk 1024#--max-context 203136 --prefill-chunk 1280#' \
       -e 's#--drafter /opt/models/dl2/assistant.q8full.ninferno#--drafter /opt/models/dl2/assistant.nvfp4.ninferno#' \
       -e 's#FP8 rings, 202K - fast#FP8 rings, 203K - fast#' \
       -e '/^Environment=NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64$/a Environment=NINFERNO_GEMMA4_SKIP_LAST_LAYER=1' $U
systemctl daemon-reload
diff $BAK $U || true

# No request in flight: the last req# line in the journal must be a "done" (or none at all),
# and the unit quiet for 20 s.
while :; do
    last=$(journalctl -u gemma4-31b-fast --since "-30 min" --no-pager -o cat | grep -E "req#[0-9]+ (started|done|failed|cancel)" | tail -1)
    recent=$(journalctl -u gemma4-31b-fast --since "-20 s" --no-pager -o cat | grep -c "req#" || true)
    if ! echo "$last" | grep -q " started " && [ "$recent" = 0 ]; then break; fi
    echo "$(date +%T) waiting: request in flight"; sleep 5
done
echo "$(date -u +%T) UTC restarting (last request line: ${last:0:80})"
systemctl restart gemma4-31b-fast
t0=$(date +%s)
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null && break; sleep 1; done
echo "answering after $(( $(date +%s) - t0 )) s: $(systemctl is-active gemma4-31b-fast)"
journalctl -u gemma4-31b-fast --since "-6 min" --no-pager -o cat | grep -E "context cache|plan \| rung|capacity \||speculation \||listening|Engine failed" | tail -7 | cut -c1-200
