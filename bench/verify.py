#!/usr/bin/env python3
"""verify: end-to-end check of a deployed server - models, chat, streaming, prefix reuse."""
import json, sys, time, urllib.request
sys.path.insert(0, "/root/gbench")
from gbench import blob, post

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
models = json.load(urllib.request.urlopen(BASE + "/v1/models"))
print("models", [m["id"] for m in models["data"]])

out, wall = post(BASE, {"model": "x", "max_tokens": 64, "temperature": 0,
                        "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]})
print("chat", repr(out["choices"][0]["message"]["content"]), f"{wall:.2f}s")

req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(
    {"model": "x", "max_tokens": 128, "stream": True,
     "messages": [{"role": "user", "content": "Name three rivers in Europe."}]}).encode(),
    {"Content-Type": "application/json"})
t, first, chunks, done = time.time(), None, 0, False
with urllib.request.urlopen(req) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        if line == "data: [DONE]":
            done = True
            break
        chunks += 1
        first = first or time.time() - t
print(f"stream chunks {chunks} done {done} first {first:.3f}s total {time.time() - t:.2f}s")

# Agent-style turns: a long fixed system prompt, then a growing conversation.
doc = blob()[: int(sys.argv[2]) if len(sys.argv) > 2 else 80000]
msgs = [{"role": "system", "content": "You are a research assistant. Reference document:\n" + doc},
        {"role": "user", "content": "In one sentence, what is this document about?"}]
follow = ["Name one detail from the first section.", "And one from the last section?", "Thanks."]
for turn in range(3):
    out, wall = post(BASE, {"model": "x", "max_tokens": 96, "temperature": 0, "messages": msgs})
    tm = out.get("timings") or {}
    text = out["choices"][0]["message"]["content"] or ""
    print("turn %d: prompt %s cache %s prompt_ms %s wall %.2fs decode %s" % (
        turn, tm.get("prompt_n"), tm.get("cache_n"), tm.get("prompt_ms"), wall,
        tm.get("predicted_per_second")))
    msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": follow[turn]}]
