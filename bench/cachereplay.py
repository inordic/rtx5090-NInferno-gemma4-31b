"""cachereplay: a compressed agent day in synthetic prompts, to compare context-cache settings.

Scheduled jobs start new sessions with a long system prompt and a tool set (A: ~17K-token system
text + 20 tools; B: another long head sharing part of A's text + 20 tools; C/D: short heads with 4
and 8 tools), and an interactive session grows past 60K tokens between them. The shape follows
the counts in our serve log; every text is generated here.

usage: cachereplay.py --label L [--slots N] [--out FILE]
Writes one JSON line per request (kind, prompt and cached tokens, seconds) and a summary line."""
import argparse
import json
import random
import time

from agentsynth import Client, system_text, text, tools

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--slots", type=int, default=12)
ap.add_argument("--base", default="http://127.0.0.1:8080")
ap.add_argument("--out", default="/root/gbench/results_cachereplay.jsonl")
a = ap.parse_args()

c = Client(a.base)
rng = random.Random(7)
ASK = "Reply with one short sentence and do not call any tool."
SYS_A = system_text(101, 13000)
SYS_B = system_text(102, 15000, shared_with=101, shared_frac=0.6)
SYS_C = system_text(103, 4500)
SYS_D = system_text(104, 5000)
T20, T4, T8 = tools(201, 20, 260), tools(202, 4, 260), tools(203, 8, 260)
T21 = T20 + tools(204, 1, 260)
rows = []
t0 = time.time()


def send(kind, messages, tool_list):
    r = c.chat(messages, tool_list, max_tokens=24)
    u = r["usage"]
    row = {"label": a.label, "kind": kind, "prompt": u.get("prompt_tokens", 0),
           "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
           "seconds": round(r["seconds"], 3), "messages": len(messages)}
    rows.append(row)
    print(json.dumps(row), flush=True)
    return r["content"]


def session(kind, sys_text, tool_list, turns, first_words, turn_words):
    m = [{"role": "system", "content": sys_text},
         {"role": "user", "content": f"Scheduled run {len(rows)}. " + text(rng, first_words) + " " + ASK}]
    m.append({"role": "assistant", "content": send(kind + "-new", m, tool_list)})
    for _ in range(turns - 1):
        m.append({"role": "user", "content": text(rng, turn_words) + " " + ASK})
        m.append({"role": "assistant", "content": send(kind + "-turn", m, tool_list)})


inter = [{"role": "system", "content": SYS_A}]


def interactive(turns):
    for _ in range(turns):
        inter.append({"role": "user", "content": text(rng, rng.randint(1500, 3500)) + " " + ASK})
        inter.append({"role": "assistant", "content": send("inter", inter, T21)})


for slot in range(a.slots):
    session("A", SYS_A, T20, 3, 120, 900)
    if slot % 2 == 1:
        session("B", SYS_B, T20, 2, 120, 900)
        interactive(3)
    if slot % 6 == 5:
        session("C", SYS_C, T4, 2, 80, 600)
        session("D", SYS_D, T8, 2, 80, 600)

new = [r for r in rows if r["kind"].endswith("-new")]
summary = {"label": a.label, "summary": True, "requests": len(rows),
           "total_s": round(sum(r["seconds"] for r in rows), 1),
           "new_s": round(sum(r["seconds"] for r in new), 1),
           "new_mean_s": round(sum(r["seconds"] for r in new) / max(1, len(new)), 2),
           "new_cached_frac": round(sum(r["cached"] for r in new) / max(1, sum(r["prompt"] for r in new)), 3),
           "all_cached_frac": round(sum(r["cached"] for r in rows) / max(1, sum(r["prompt"] for r in rows)), 3),
           "wall_s": round(time.time() - t0, 1)}
for k in ("A", "B", "C", "D"):
    ks = [r for r in new if r["kind"] == k + "-new"]
    summary[f"{k}_new_mean_s"] = round(sum(r["seconds"] for r in ks) / max(1, len(ks)), 2)
it = [r for r in rows if r["kind"] == "inter"]
summary["inter_mean_s"] = round(sum(r["seconds"] for r in it) / max(1, len(it)), 2)
print(json.dumps(summary), flush=True)
with open(a.out, "a") as f:
    for r in rows + [summary]:
        f.write(json.dumps(r) + "\n")
