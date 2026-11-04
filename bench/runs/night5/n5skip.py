#!/usr/bin/env python3
"""n5skip LOG TOKENS...: on the counting build, the share of D = 512 score tiles whose best score sits
more than 2^-6 .. 2^-14 below the running maximum, per cold prompt of natural text."""
import json, random, re, sys, time, urllib.request
sys.path.insert(0, "/root/gbench")
from gbench import blob

log, sizes = sys.argv[1], [int(x) for x in sys.argv[2:]]


def counts():
    last = None
    for line in open(log, errors="replace"):
        if "skipcount" in line:
            last = line
    return [int(x) for x in re.findall(r"\d+", last.split("skipcount", 1)[1])] if last else [0] * 6


text, rnd = blob(), random.Random(9)
for n in sizes:
    off = rnd.randrange(0, len(text) - int(n * 3.9) - 1)
    body = {"model": "x", "max_tokens": 1, "temperature": 0, "messages": [
        {"role": "user", "content": f"[skip {time.time_ns()}]\n" + text[off:off + int(n * 3.9)] + "\n\nWhat is this about?"}]}
    before = counts()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        r.read()
    time.sleep(1)
    after = counts()
    d = [a - b for a, b in zip(after, before)]
    share = " ".join(f"2^-{t}={100 * c / max(1, d[0]):.1f}%" for t, c in zip((6, 8, 10, 12, 14), d[1:]))
    print(f"SKIP {n}: tiles {d[0]} {share}", flush=True)
