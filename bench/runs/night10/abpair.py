"""abpair.py [A_LABELS] [B_LABELS]: drafter A/B from heldout10.py's per-request rows (n10/heldout-LABEL.json).
Defaults: A = ab-n9a,ab-n9b, B = ab-n10a,ab-n10b (alternating runs, so each pair of runs also gives the noise).
Per group: decode tok/s and drafts accepted for every run, the mean of each side, and B - A. heldout10.py's own
'english' line is the English chat category only (the aggregate's name clashed with it), so the English aggregate here
is every category but Swedish and oasst2, as intended.
Paired: per request, accepted / drafted averaged over a side's runs; requests with at least 10 draft rounds in every
run; B better / worse, two-sided sign test."""
import json, math, sys

A = (sys.argv[1] if len(sys.argv) > 1 else "ab-n9a,ab-n9b").split(",")
B = (sys.argv[2] if len(sys.argv) > 2 else "ab-n10a,ab-n10b").split(",")
NOT_EN = ("swedish", "oasst2")
runs = {l: json.load(open(f"/root/gbench/n10/heldout-{l}.json"))["rows"] for l in A + B}


def summ(rs):
    out = sum(r["out"] for r in rs)
    secs = sum(r["out"] / r["decode"] for r in rs if r["decode"] > 0)
    return (out / secs if secs else 0.0, 100 * sum(r["acc"] for r in rs) / max(1, sum(r["drafted"] for r in rs)), len(rs))


GROUPS = [("all", lambda r: True), ("english (all but sv/oasst2)", lambda r: r["cat"] not in NOT_EN),
          ("other languages", lambda r: r["cat"] in NOT_EN), ("greedy", lambda r: r["temp"] == 0),
          ("sampled", lambda r: r["temp"] > 0)]
cats = sorted({r["cat"] for rs in runs.values() for r in rs})
GROUPS += [(c, (lambda c: lambda r: r["cat"] == c)(c)) for c in cats]

print(f"{'group':30s} " + " ".join(f"{l:>15s}" for l in A + B) + f" {'A mean':>13s} {'B mean':>13s} {'B-A tok/s':>9s}")
for name, f in GROUPS:
    s = {l: summ([r for r in runs[l] if f(r)]) for l in A + B}
    am = [sum(s[l][i] for l in A) / len(A) for i in (0, 1)]
    bm = [sum(s[l][i] for l in B) / len(B) for i in (0, 1)]
    print(f"{name + ' (' + str(s[A[0]][2]) + ')':30s} " + " ".join(f"{s[l][0]:7.1f}/{s[l][1]:5.1f}% " for l in A + B)
          + f" {am[0]:6.1f}/{am[1]:4.1f}% {bm[0]:6.1f}/{bm[1]:4.1f}% {bm[0] - am[0]:+8.1f}")


def binom_p(k, n):
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


K = None
for kind, sel in (("greedy", lambda t: t == 0), ("sampled", lambda t: t > 0), ("all", lambda t: True)):
    for sub, cf in (("", lambda c: True), (" english", lambda c: c not in NOT_EN)):
        by = {}
        for l in A + B:
            for r in runs[l]:
                if sel(r["temp"]) and cf(r["cat"]):
                    by.setdefault(r["id"], {})[l] = r
        better = worse = same = 0
        for rid, d in by.items():
            if len(d) != len(A) + len(B) or any(d[l]["drafted"] < 70 for l in d):
                continue
            a = sum(d[l]["acc"] / d[l]["drafted"] for l in A) / len(A)
            b = sum(d[l]["acc"] / d[l]["drafted"] for l in B) / len(B)
            if b > a: better += 1
            elif b < a: worse += 1
            else: same += 1
        print(f"paired {kind}{sub}: B accepts more on {better}, fewer on {worse}, same {same}; sign test p = {binom_p(better, better + worse):.3g}")

for c in cats:
    by = {}
    for l in A + B:
        for r in runs[l]:
            if r["cat"] == c:
                by.setdefault(r["id"], {})[l] = r
    better = worse = same = 0
    for rid, d in by.items():
        if len(d) != len(A) + len(B) or any(d[l]["drafted"] < 70 for l in d):
            continue
        a = sum(d[l]["acc"] / d[l]["drafted"] for l in A) / len(A)
        b = sum(d[l]["acc"] / d[l]["drafted"] for l in B) / len(B)
        better += b > a; worse += b < a; same += b == a
    print(f"paired {c}: B accepts more on {better}, fewer on {worse}, same {same}; sign test p = {binom_p(better, better + worse):.3g}")

# run-to-run noise: same drafter, two runs
for side, ls in (("A", A), ("B", B)):
    if len(ls) == 2:
        s0, s1 = summ(runs[ls[0]]), summ(runs[ls[1]])
        print(f"noise {side} ({ls[0]} vs {ls[1]}), all: {s0[0]:.1f} vs {s1[0]:.1f} tok/s, {s0[1]:.1f} vs {s1[1]:.1f}% accepted")
