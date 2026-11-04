"""reqstats.py SINCE [UNTIL]: aggregate numbers from the fast unit's "req#N done" lines (no content)."""
import re
import statistics as st
import subprocess
import sys

args = ["journalctl", "-u", "gemma4-31b-fast", "--since", sys.argv[1], "--no-pager", "-o", "cat"]
if len(sys.argv) > 2:
    args += ["--until", sys.argv[2]]
lines = subprocess.run(args, capture_output=True, text=True).stdout.splitlines()


def num(s):
    return float(s.replace(",", ""))


def secs(s):
    m = re.match(r"(?:(\d+)m )?([\d.]+) ?(ms|s)", s)
    if not m:
        return None
    v = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    return v + 60 * int(m.group(1) or 0)


def rate(s):
    m = re.match(r"([\d.]+)(k?) tok/s", s)
    return float(m.group(1)) * (1000 if m.group(2) else 1) if m else None


rows, warns = [], 0
for ln in lines:
    if " WARN " in ln and "tool markup returned as text" in ln:
        warns += 1
    if not re.search(r"req#\d+ done", ln):
        continue
    f = {}
    for part in ln.split(" | "):
        part = part.strip()
        for key in ("prompt", "output", "cache", "TTFT", "total", "prefill", "decode", "mtp accepted"):
            if part.startswith(key + " "):
                f[key] = part[len(key) + 1:]
    r = {"prompt": num(f.get("prompt", "0")), "output": num(f.get("output", "0"))}
    c = f.get("cache", "0")
    m = re.match(r"([\d,]+)(?: \(([\d.]+)%, ([^)]+)\))?", c)
    r["cached"] = num(m.group(1)) if m else 0
    r["kind"] = (m.group(3) if m and m.group(3) else "none")
    r["ttft"] = secs(f["TTFT"]) if "TTFT" in f else None
    r["prefill"] = rate(f["prefill"]) if "prefill" in f else None
    r["decode"] = rate(f["decode"]) if "decode" in f else None
    m = re.match(r"(\d+)/(\d+)", f.get("mtp accepted", ""))
    r["acc"] = (int(m.group(1)), int(m.group(2))) if m else None
    rows.append(r)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


n = len(rows)
print(f"requests {n} | tool-markup warnings {warns}")
if not n:
    sys.exit()
P = [r["prompt"] for r in rows]
print(f"prompt tokens: median {st.median(P):,.0f} | p90 {q(P, .9):,.0f} | max {max(P):,.0f} | total {sum(P):,.0f}")
tc = sum(r["cached"] for r in rows)
print(f"cached tokens: {tc:,.0f} of {sum(P):,.0f} ({100 * tc / sum(P):.1f}%)")
kinds = {}
for r in rows:
    kinds.setdefault(r["kind"], []).append(r)
for k, rs in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
    share = [100 * r["cached"] / r["prompt"] for r in rs if r["prompt"]]
    print(f"  cache {k}: {len(rs)} requests, median hit {st.median(share):.1f}%")
hits = [100 * r["cached"] / r["prompt"] for r in rows if r["prompt"]]
print(f"  requests with >=90% cached: {sum(h >= 90 for h in hits)} | 50-90%: {sum(50 <= h < 90 for h in hits)} | <50%: {sum(h < 50 for h in hits)}")
T = [r["ttft"] for r in rows if r["ttft"] is not None]
print(f"TTFT s: median {st.median(T):.2f} | p90 {q(T, .9):.2f} | max {max(T):.1f}")
new = [(r["prompt"] - r["cached"], r["prefill"]) for r in rows if r["prefill"]]
for lo, hi in ((0, 2048), (2048, 8192), (8192, 32768), (32768, 1e9)):
    v = [p for t, p in new if lo <= t < hi]
    if v:
        print(f"prefill tok/s, {lo:>6,.0f}-{hi if hi < 1e9 else 0:>6,.0f} new tokens: n {len(v)} | median {st.median(v):,.0f}")
D = [r["decode"] for r in rows if r["decode"] and r["output"] >= 32]
print(f"decode tok/s (output >= 32): n {len(D)} | median {st.median(D):.1f} | p10 {q(D, .1):.1f} | p90 {q(D, .9):.1f}")
a = [r["acc"] for r in rows if r["acc"]]
print(f"MTP acceptance: {sum(x for x, _ in a)}/{sum(y for _, y in a)} ({100 * sum(x for x, _ in a) / max(1, sum(y for _, y in a)):.1f}%)")
O = [r["output"] for r in rows]
print(f"output tokens: median {st.median(O):,.0f} | total {sum(O):,.0f}")
for lo, hi in ((0, 32768), (32768, 65536), (65536, 1e9)):
    v = [r for r in rows if r["prefill"] and r["prompt"] - r["cached"] < 2048 and lo <= r["prompt"] < hi]
    if v:
        print(f"<2K new tokens, context {lo // 1024}K-{'' if hi >= 1e9 else int(hi) // 1024}K: n {len(v)} | "
              f"median prefill {st.median(r['prefill'] for r in v):,.0f} tok/s | median TTFT {st.median([r['ttft'] for r in v if r['ttft'] is not None] or [0]):.2f} s")
for lo, hi in ((32768, 65536), (65536, 1e9)):
    v = [r for r in rows if r["prefill"] and r["prompt"] - r["cached"] >= 32768 and lo <= r["prompt"] < hi]
    if v:
        print(f">=32K new tokens, context {lo // 1024}K-{'' if hi >= 1e9 else int(hi) // 1024}K: n {len(v)} | median prefill {st.median(r['prefill'] for r in v):,.0f} tok/s")
