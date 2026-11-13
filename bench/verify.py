#!/usr/bin/env python3
"""verify: end-to-end check of a deployed server - models, chat, streaming, prefix reuse.
Every check is asserted: the script prints VERIFY OK and exits 0 only if all pass, otherwise it prints each
failure and VERIFY FAIL and exits 1 (2 if the server could not be reached at all).
  python3 verify.py [BASE] [DOC_CHARS] [--model ID]"""
import argparse, json, sys, time, urllib.request
sys.path.insert(0, "/root/gbench")
from gbench import blob, post

ap = argparse.ArgumentParser()
ap.add_argument("base", nargs="?", default="http://127.0.0.1:8080")
ap.add_argument("doc_chars", nargs="?", type=int, default=80000)
ap.add_argument("--model", help="the model id the server must report")
a = ap.parse_args()
BASE = a.base
fails = []


def check(ok, what):
    if not ok:
        fails.append(what)
        print("  FAIL", what)
    return ok


try:
    models = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))
except Exception as e:
    print("VERIFY FAIL: server not reachable:", e)
    sys.exit(2)
ids = [m["id"] for m in models.get("data", [])]
print("models", ids)
check(len(ids) >= 1, "no model listed")
if a.model:
    check(a.model in ids, f"model id {a.model!r} not served (got {ids})")

out, wall = post(BASE, {"model": "x", "max_tokens": 64, "temperature": 0,
                        "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]})
ans = (out["choices"][0]["message"]["content"] or "").strip()
print("chat", repr(ans), f"{wall:.2f}s")
check(ans.rstrip(".") == "391", f"17 * 23 answered {ans!r}, want '391'")

req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(
    {"model": "x", "max_tokens": 128, "stream": True,
     "messages": [{"role": "user", "content": "Name three rivers in Europe."}]}).encode(),
    {"Content-Type": "application/json"})
t, first, chunks, done, text = time.time(), None, 0, False, ""
with urllib.request.urlopen(req, timeout=120) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        if line == "data: [DONE]":
            done = True
            break
        chunks += 1
        first = first or time.time() - t
        try:
            text += json.loads(line[5:])["choices"][0].get("delta", {}).get("content") or ""
        except Exception:
            check(False, f"stream chunk not JSON: {line[:80]!r}")
print(f"stream chunks {chunks} done {done} first {first or 0:.3f}s total {time.time() - t:.2f}s")
check(done, "stream ended without [DONE]")
check(chunks >= 3 and len(text.strip()) > 10, f"stream too short ({chunks} chunks, {len(text)} chars)")

# Agent-style turns: a long fixed system prompt, then a growing conversation. Every turn after the first must
# reuse the cached prefix: cache_n at least 90% of the previous turn's whole prompt (its new plus cached tokens).
doc = blob()[: a.doc_chars]
msgs = [{"role": "system", "content": "You are a research assistant. Reference document:\n" + doc},
        {"role": "user", "content": "In one sentence, what is this document about?"}]
follow = ["Name one detail from the first section.", "And one from the last section?", "Thanks."]
prev = None
for turn in range(3):
    out, wall = post(BASE, {"model": "x", "max_tokens": 96, "temperature": 0, "messages": msgs})
    tm = out.get("timings") or {}
    text = out["choices"][0]["message"]["content"] or ""
    print("turn %d: prompt %s cache %s prompt_ms %s wall %.2fs decode %s" % (
        turn, tm.get("prompt_n"), tm.get("cache_n"), tm.get("prompt_ms"), wall,
        tm.get("predicted_per_second")))
    check(len(text.strip()) > 0, f"turn {turn}: empty reply")
    if turn > 0:
        cache_n = tm.get("cache_n") or 0
        check(prev is not None and cache_n >= 0.9 * prev, f"turn {turn}: cache {cache_n} of a {prev}-token prefix")
    prev = (tm.get("prompt_n") or 0) + (tm.get("cache_n") or 0)   # prompt_n counts only the new tokens
    msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": follow[turn]}]

if fails:
    print(f"VERIFY FAIL: {len(fails)} check(s): " + "; ".join(fails))
    sys.exit(1)
print("VERIFY OK")
