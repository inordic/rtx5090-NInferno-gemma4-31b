#!/bin/bash
# n2f_speed: night 2, batch F - speed levers from the research, on the production config:
#  F1 split-K combine batch 4 (apps-final6) against 24 (apps-final5): greedy output identity, decode
#     at short and 32K context, alternated A B A B;
#  F2 the wide prefill tile from T >= 64 (env) on cached agent turns and a 2K prompt;
#  F3 shared values 1 (K + rotated V stored) against 2 (K only) for decode at 32K / 64K of context;
#  F4 the per-request overhead: --request-log-jsonl phases of tiny requests on an idle server.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
until grep -q "N2A DONE" n2a_context.log 2>/dev/null; do sleep 30; done
./idle_gate_any.sh 5
step "idle: taking the GPU"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
QN=/opt/models/dl2/qat-nvfp4.r31.ninferno
DQ=/opt/models/dl2/qat-assistant.q40qat.ninferno
export NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1 NINFERNO_GEMMA4_SHARED_VALUES=2
sed "s#^S=.*#S=/opt/NInferno/build/apps-final6/ninferno-serve#" srv.sh > srv_final6.sh; chmod +x srv_final6.sh
greedy() {  # greedy <tag>: three fixed prompts at temperature 0, texts to greedy_<tag>.json
    python3 - "$1" <<"PY"
import json, sys, urllib.request
out = []
for p in ["Explain how a hash map handles collisions.", "Write a Python function that merges two sorted lists.",
          "Summarize the causes of the French Revolution in five bullet points."]:
    body = {"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": 200, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})))
    out.append(r["choices"][0]["message"]["content"])
json.dump(out, open(f"/root/gbench/greedy_{sys.argv[1]}.json", "w"))
PY
}
run() {  # run <srv> <label> [args]: production config
    local srv=$1 label=$2; shift 2
    ./$srv start $label $QN --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 --spec mtp --drafter $DQ --draft-tokens 7 "$@"
}
for rep in 1 2; do
    for v in final5 final6; do
        step "F1 rep $rep: $v"
        run srv_$v.sh F1-$v-$rep && greedy $v-$rep && \
            python3 gbench.py --label "F1 $v rep$rep" --pp 2048 --tg 2048,8192,32768 --reps 1
        ./srv.sh stop
    done
done
python3 -c "
import json
a, b = json.load(open('/root/gbench/greedy_final5-1.json')), json.load(open('/root/gbench/greedy_final6-1.json'))
print('F1 greedy identical:', a == b, [x == y for x, y in zip(a, b)])"
for m in 1024 64; do
    step "F2 wide prefill from T >= $m"
    NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=$m run srv_final6.sh F2-w$m && \
        python3 turnprobe.py http://127.0.0.1:8080 && \
        python3 gbench.py --label "F2 wide-min-$m" --pp 2048,8192 --tg "" --skip-chat --reps 2
    ./srv.sh stop
done
for sv in 2 1; do
    step "F3 shared values $sv: decode at 32K / 64K (90K context)"
    NINFERNO_GEMMA4_SHARED_VALUES=$sv ./srv_final6.sh start F3-sv$sv $QN --max-context 90000 --prefill-chunk 1024 --kv-dtype fp8 \
        --spec mtp --drafter $DQ --draft-tokens 7 && \
        python3 gbench.py --label "F3 shared-values-$sv" --pp "" --tg 32768,65536 --skip-chat --reps 1
    ./srv.sh stop
done
step "F4 request phases on an idle server"
rm -f /root/gbench/reqlog.jsonl
run srv_final6.sh F4-reqlog --request-log-jsonl /root/gbench/reqlog.jsonl && \
    python3 - <<"PY"
import json, time, urllib.request
for i in range(10):
    body = {"model": "x", "messages": [{"role": "user", "content": f"Say the number {i}."}], "max_tokens": 4, "temperature": 0}
    t = time.time()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})).read()
    print(f"request {i}: {1000 * (time.time() - t):.0f} ms")
    time.sleep(1)
PY
./srv.sh stop
python3 -c "
import json
rows = [json.loads(l) for l in open('/root/gbench/reqlog.jsonl')]
print('F4 keys:', sorted(rows[-1].keys()))
for r in rows[-3:]: print(json.dumps({k: v for k, v in r.items() if any(s in k for s in ('ms', 'time', 'queue', 'ttft', 'phase', 'prefill', 'tokens'))})[:600])"
step "restoring the fast unit"
./deploy_fast.sh
step "N2F DONE"
