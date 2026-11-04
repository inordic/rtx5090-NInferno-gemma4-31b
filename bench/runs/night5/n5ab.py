#!/usr/bin/env python3
"""n5ab: greedy answers over fixed long prompts, to compare two builds or switches token for token.
  n5ab.py run OUT.json     per prompt: a cold request, the same prompt again (the zero-suffix path),
                           and a follow-up turn on its answer (the checkpoint path)
  n5ab.py cmp A.json B.json   how many answers differ
The server must be fresh, so the first request of each prompt finds no cache."""
import json, random, sys, urllib.request
sys.path.insert(0, "/root/gbench")
from gbench import blob

BASE = "http://127.0.0.1:8080"
TOKENS = [3000, 6000, 9000, 12000, 17000, 24000, 33000, 48000, 64000, 100000]
CPT = 3.9


def ask(messages):
    body = {"model": "x", "messages": messages, "max_tokens": 48, "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    m = out["choices"][0]
    return {"text": m["message"].get("content") or "", "finish": m.get("finish_reason"),
            "usage": out.get("usage") or {}}


def run(path):
    text, rnd, rows = blob(), random.Random(5), []
    for n in TOKENS:
        off = rnd.randrange(0, len(text) - int(n * CPT) - 1)
        user = f"[prompt {n}]\n" + text[off:off + int(n * CPT)] + "\n\nSummarize the text above in two sentences."
        first = [{"role": "user", "content": user}]
        cold = ask(first)
        again = ask(first)
        follow = ask(first + [{"role": "assistant", "content": cold["text"]},
                              {"role": "user", "content": "Name the main topic in three words."}])
        rows.append({"tokens": n, "cold": cold, "again": again, "follow": follow})
        print(n, repr(cold["text"][:60]), flush=True)
    json.dump(rows, open(path, "w"))


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same = total = 0
    for x, y in zip(A, B):
        for k in ("cold", "again", "follow"):
            total += 1
            if x[k]["text"] == y[k]["text"]:
                same += 1
            else:
                print(f"differs: {x['tokens']} {k}: {x[k]['text'][:70]!r} | {y[k]['text'][:70]!r}")
    print(f"same {same}/{total}: {a} {b}")


if sys.argv[1] == "run":
    run(sys.argv[2])
else:
    cmp(sys.argv[2], sys.argv[3])
