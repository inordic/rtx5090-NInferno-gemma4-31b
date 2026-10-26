#!/usr/bin/env python3
"""turnprobe: latency of agent-style turns on a cached prefix - a long system prompt, then turns
that append N new tokens each; prints the server's timings per turn (prompt_ms, cache_n, TTFT)."""
import json, sys, time, urllib.request
sys.path.insert(0, "/root/gbench")
from gbench import blob

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
doc = blob()
filler = doc[200000:400000]

def turn(msgs, max_tokens=8):
    body = {"model": "x", "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time(); first = None; text = ""; timings = {}
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            for c in ev.get("choices") or []:
                d = (c.get("delta") or {}).get("content")
                if d:
                    first = first or time.time() - t
                    text += d
            timings = ev.get("timings") or timings
    return first, time.time() - t, text, timings

msgs = [{"role": "system", "content": "You are a research assistant. Reference:\n" + doc[:70000]},
        {"role": "user", "content": "Reply with OK."}]
f, w, text, tm = turn(msgs)
print(f"base: prompt {tm.get('prompt_n')} cache {tm.get('cache_n')} prompt_ms {tm.get('prompt_ms')} ttft {f:.3f}s wall {w:.2f}s")
pos = 0
for n_chars in (60, 250, 500, 1000, 2000, 4000):
    msgs += [{"role": "assistant", "content": text or "OK."},
             {"role": "user", "content": "Tool result:\n" + filler[pos:pos + n_chars] + "\nReply with OK."}]
    pos += n_chars
    f, w, text, tm = turn(msgs)
    print(f"+{n_chars:5d} chars: prompt {tm.get('prompt_n')} cache {tm.get('cache_n')} "
          f"new {(tm.get('prompt_n') or 0) - (tm.get('cache_n') or 0)} prompt_ms {tm.get('prompt_ms')} "
          f"ttft {f if f is None else round(f, 3)}s wall {w:.2f}s")
