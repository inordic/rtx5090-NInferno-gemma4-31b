"""livecache: aggregate prompt-cache use of the fast unit from its serve log, for two windows.

usage: journalctl -u gemma4-31b-fast -o cat --since ... | python3 livecache.py SPLIT_UTC [EXCLUDE_FROM EXCLUDE_TO] [--json]
Requests started before SPLIT_UTC count as 'before', the rest as 'after'; requests started inside
[EXCLUDE_FROM, EXCLUDE_TO) (our own checks) are left out. Prints aggregates only; --json prints them as JSON lines instead."""
import json
import re
import statistics
import sys

START = re.compile(r"^(\S+ \S+)\s+INFO\s+req#(\d+) started \| (\S+) (stream|non-stream) \| request \S+ \| (\d+) messages? \| "
                   r"max output [\d,]+ \| thinking (\w+)(?: \| tools (\d+))?")
DONE = re.compile(r"^(\S+ \S+)\s+INFO\s+req#(\d+) done \| \S+ \| ([^|]+?) \| prompt ([\d,]+) \| output ([\d,]+) \| "
                  r"cache ([\d,]+) \(([\d.]+)%(?:, ([^)]+))?\) \| TTFT ([\d.]+) ?(ms|s) \| total ([\d.]+) ?(ms|s)")
FAIL = re.compile(r"req#(\d+) (failed|cancel)|Engine failed")

as_json = "--json" in sys.argv
args = [a for a in sys.argv[1:] if a != "--json"]
split = args[0]
ex_from, ex_to = (args[1], args[2]) if len(args) > 2 else ("9", "9")
num = lambda s: int(s.replace(",", ""))
sec = lambda v, u: float(v) / 1000 if u == "ms" else float(v)
open_reqs, rows, fails = {}, [], {"before": 0, "after": 0}
last_t = "0"
for line in sys.stdin:
    if line[:2] == "20":
        last_t = line[:23]
    m = START.match(line)
    if m:
        open_reqs[m.group(2)] = {"t": m.group(1), "msgs": int(m.group(5)), "tools": int(m.group(7) or 0)}
        continue
    m = DONE.match(line)
    if m and m.group(2) in open_reqs:
        r = open_reqs.pop(m.group(2))
        r.update(prompt=num(m.group(4)), cache=num(m.group(6)), path=m.group(8) or "none",
                 ttft=sec(m.group(9), m.group(10)), total=sec(m.group(11), m.group(12)))
        rows.append(r)
        continue
    if FAIL.search(line):  # an engine exit's own line carries no timestamp: take the last one seen
        fails["before" if last_t < split else "after"] += 1


def summary(win, name, rs):
    if not rs:
        if not as_json:
            print(f"  {name}: none")
        return
    p, c = sum(r["prompt"] for r in rs), sum(r["cache"] for r in rs)
    hits = sum(1 for r in rs if r["cache"] >= 0.5 * r["prompt"])
    tt = [r["ttft"] for r in rs]
    paths = {}
    for r in rs:
        paths[r["path"]] = paths.get(r["path"], 0) + 1
    if as_json:
        print(json.dumps({"window": win, "group": name.strip(), "requests": len(rs), "mean_prompt": round(p / len(rs)),
                          "cached_frac": round(c / max(p, 1), 3), "half_cached": hits,
                          "ttft_mean_s": round(statistics.mean(tt), 2), "ttft_median_s": round(statistics.median(tt), 2),
                          "ttft_p90_s": round(sorted(tt)[int(0.9 * (len(tt) - 1))], 2), "reuse_paths": paths}))
        return
    print(f"  {name}: {len(rs)} requests | mean prompt {p / len(rs) / 1000:.1f}K | cached {c / max(p, 1):.1%} of prompt tokens | "
          f"half-or-more cached {hits}/{len(rs)} | TTFT mean {statistics.mean(tt):.2f}s median {statistics.median(tt):.2f}s "
          f"p90 {sorted(tt)[int(0.9 * (len(tt) - 1))]:.2f}s | prefill time {sum(tt) / 60:.1f} min")
    print(f"      reuse paths: " + ", ".join(f"{k} {v}" for k, v in sorted(paths.items(), key=lambda kv: -kv[1])))


for win in ("before", "after"):
    rs = [r for r in rows if (r["t"] < split) == (win == "before") and not (ex_from <= r["t"] < ex_to)]
    if rs:
        span = f"{rs[0]['t'][:16]} .. {rs[-1]['t'][:16]} UTC"
        print(json.dumps({"window": win, "span": span, "failure_lines": fails[win]}) if as_json
              else f"{win}: {span}, failure lines {fails[win]}")
    summary(win, "all", rs)
    summary(win, "new sessions with tools (2 messages)", [r for r in rs if r["msgs"] == 2 and r["tools"]])
    summary(win, "  of them heads >= 20K tokens", [r for r in rs if r["msgs"] == 2 and r["tools"] and r["prompt"] >= 20000])
    summary(win, "  of them heads < 20K tokens", [r for r in rs if r["msgs"] == 2 and r["tools"] and r["prompt"] < 20000])
    summary(win, "continuing turns (3+ messages)", [r for r in rs if r["msgs"] > 2])
