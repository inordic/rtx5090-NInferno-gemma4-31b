#!/usr/bin/env python3
"""gbench: PP / TG / short-chat bench for an OpenAI-compatible NInferno server.

PP   : unique-nonce prompts of ~2K/8K/24K tokens, max_tokens=1, server prompt_per_second (median).
TG   : decode tok/s (server predicted_per_second) after a 2K / 8K context, 512 out tokens.
CHAT : 10 short diverse prompts, 512 out tokens, aggregate decode tok/s at C=1.
Each TG/CHAT is run greedy (t=0) and with model-default sampling (t=1, top_k 64).
"""
import argparse, json, statistics, sys, time, urllib.request, random

HAY = "/opt/NInferno/examples/cli/messages/long_niah_256k.json"
CHAT = [
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "Write a SQL query that returns the top 3 customers by total order value per country, with table schemas you assume.",
    "Explain how TCP congestion control works, covering slow start, congestion avoidance and fast recovery.",
    "Write a short story (about 300 words) about a lighthouse keeper who discovers a message in a bottle.",
    "Translate into German and French: 'The meeting has been moved to Thursday afternoon because the client is travelling.' Then explain any tricky grammar.",
    "A train leaves at 09:40 and travels 315 km at an average of 84 km/h. It stops twice for 7 minutes. When does it arrive? Show your reasoning step by step.",
    "Write a bash script that finds the 10 largest files under a directory, prints human-readable sizes, and skips .git folders.",
    "Compare PostgreSQL and MongoDB for an e-commerce product catalog. Give a recommendation with pros and cons.",
    "Implement a thread-safe LRU cache in TypeScript with get/put and a max size, with comments.",
    "Summarize the causes and consequences of the 2008 financial crisis in a structured list.",
]
QUESTION = "\n\nSummarize the document above section by section, in detail."


def post(base, body, timeout=3600):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.time() - t


def chat(base, prompt, max_tokens, greedy=True, extra=None):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if greedy:
        body["temperature"] = 0.0
    if extra:
        body.update(extra)
    out, wall = post(base, body)
    tm = out.get("timings") or {}
    return {"pp": tm.get("prompt_per_second"), "pn": tm.get("prompt_n"), "cache": tm.get("cache_n"),
            "tg": tm.get("predicted_per_second"), "tn": tm.get("predicted_n"),
            "tms": tm.get("predicted_ms"), "pms": tm.get("prompt_ms"), "wall": wall,
            "spec": tm.get("speculation") or {}, "text": out["choices"][0]["message"].get("content") or ""}


def blob():
    user = json.load(open(HAY))[-1]["content"]
    return user.split("<document>\n", 1)[1].split("\n</document>", 1)[0]


def med(xs):
    xs = [x for x in xs if x]
    return round(statistics.median(xs), 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="/root/gbench/results.jsonl")
    ap.add_argument("--pp", default="2048,8192,24576")
    ap.add_argument("--tg", default="2048,8192")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--skip-chat", action="store_true")
    ap.add_argument("--skip-sampled", action="store_true")
    ap.add_argument("--chat-n", type=int, default=10)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    text = blob()
    cpt = 3.9  # chars per token, approx for this corpus
    row = {"label": a.label, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    chat(base, "Say hello in one short sentence.", 16)  # warm-up
    rnd = random.Random(1234)

    # PP
    for n in [int(x) for x in a.pp.split(",") if x]:
        vals = []
        for r in range(a.reps):
            off = rnd.randrange(0, len(text) - int(n * cpt) - 1)
            p = f"[run {time.time_ns()}]\n" + text[off:off + int(n * cpt)] + QUESTION
            res = chat(base, p, 1)
            vals.append(res["pp"])
            row[f"pp{n}_ntok"] = res["pn"]
        row[f"pp{n}"] = med(vals)
        print(f"PP {n}: {row[f'pp{n}']} tok/s (n={row[f'pp{n}_ntok']}) {vals}", flush=True)

    # TG after context
    modes = [("g", True)] + ([] if a.skip_sampled else [("s", False)])
    for n in [int(x) for x in a.tg.split(",") if x]:
        for tag, greedy in modes:
            vals, rounds = [], []
            for r in range(a.reps):
                off = rnd.randrange(0, len(text) - int(n * cpt) - 1)
                p = f"[run {time.time_ns()}]\n" + text[off:off + int(n * cpt)] + QUESTION
                res = chat(base, p, 512, greedy)
                vals.append(res["tg"])
                rounds.append(res["spec"].get("config"))
            row[f"tg{n}{tag}"] = med(vals)
            print(f"TG@{n} {'greedy' if greedy else 'sampled'}: {row[f'tg{n}{tag}']} tok/s {vals} spec={rounds}", flush=True)

    # Short chat C=1
    if not a.skip_chat:
        for tag, greedy in modes:
            tok, ms, specs = 0, 0.0, []
            for p in CHAT[:a.chat_n]:
                res = chat(base, p, 512, greedy)
                tok += res["tn"] or 0
                ms += res["tms"] or 0
                specs.append(res["spec"])
            row[f"chat{tag}"] = round(tok / (ms / 1000), 1) if ms else None
            print(f"CHAT C=1 {'greedy' if greedy else 'sampled'}: {row[f'chat{tag}']} tok/s ({tok} tok) spec={specs[0].get('config')}", flush=True)
            if specs and specs[0]:
                row[f"chat{tag}_spec"] = {k: specs[0].get(k) for k in ("config", "fallback", "reason")}

    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
