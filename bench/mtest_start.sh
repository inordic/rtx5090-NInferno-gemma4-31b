#!/bin/bash
# mtest_start.sh: behind a 3-minute idle gate, swap the fast unit for apps-metrics (1a3d564 + request counters)
# with the same flags on 8080, then send an API request, a chat through the test UI (:8091) and a tool call.
cd /root/gbench
step() { echo "=== $(date +%T) $*"; }
./idle_gate_any.sh 3
step "idle: swapping in apps-metrics"
systemctl stop gemma4-31b-fast
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/v1/models > /dev/null || break; sleep 1; done
export NINFERNO_GEMMA4_SHARED_VALUES=2 NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1
sed "s#^S=.*#S=/opt/NInferno/build/apps-metrics/ninferno-serve#" srv.sh > srv_metrics.sh; chmod +x srv_metrics.sh
./srv_metrics.sh start gemma4-31b-fast /opt/models/dl2/qat-nvfp4.r31.ninferno --max-context 155000 --prefill-chunk 1024 --kv-dtype fp8 \
    --spec mtp --drafter /opt/models/dl2/qat-assistant.q40qat.ninferno --draft-tokens 7 | tail -2
step "API request (non-stream)"
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"max_tokens\":300,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Explain what a hash map is in three paragraphs.\"}]}" | python3 -c "import json,sys; u=json.load(sys.stdin)[\"usage\"]; print(\"api usage\", u)"
sleep 2
step "chat through the test UI (stream)"
curl -sN http://127.0.0.1:8091/v1/chat/completions -H "Content-Type: application/json" -d "{\"stream\":true,\"stream_options\":{\"include_usage\":true},\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"Name five rivers in Europe and one fact about each.\"}]}" | grep -c "^data:"
sleep 2
step "API tool call (stream)"
curl -sN http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"stream\":true,\"max_tokens\":200,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Oslo? Use the tool.\"}]}" | grep -c "^data:"
sleep 12
step "engine /metrics (new counters)"
curl -s http://127.0.0.1:8080/metrics | grep -E "^ninferno_(request|speculative|requests_finished)" 
step "test UI engine snapshot"
curl -s http://127.0.0.1:8091/metrics | python3 -c "
import json,sys
e=json.load(sys.stdin)[\"engine\"]
print(\"per_request\", e[\"per_request\"], \"totals\", json.dumps(e[\"totals\"]))
for r in e[\"requests\"]: print(\"row\", json.dumps({k: r[k] for k in (\"source\",\"count\",\"finish\",\"prompt_tokens\",\"cached_tokens\",\"output_tokens\",\"prefill_tok_s\",\"decode_tok_s\",\"e2e_s\",\"queue_s\")}))"
step "MTEST STARTED"
