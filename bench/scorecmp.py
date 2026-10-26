#!/usr/bin/env python3
"""Compare two `ninferno-gemma4 score --dump` files: perplexity of each and top-1 agreement.

A dump is the FP32 log-probabilities of the scored tokens followed by the I32 argmax ids.
"""
import math
import struct

for w in (4096, 32768):
    out = {}
    for v in ("mode2", "i8"):
        raw = open(f"/root/gbench/score_{v}_{w}.bin", "rb").read()
        n = len(raw) // 8
        lp = struct.unpack(f"<{n}f", raw[:4 * n])
        am = struct.unpack(f"<{n}i", raw[4 * n:])
        out[v] = (lp, am)
    a, b = out["mode2"], out["i8"]
    n = min(len(a[0]), len(b[0]))
    ppl = {v: math.exp(-sum(out[v][0][:n]) / n) for v in out}
    agree = sum(1 for i in range(n) if a[1][i] == b[1][i]) / n
    dnll = sum(abs(a[0][i] - b[0][i]) for i in range(n)) / n
    print(f"SCORECMP window {w}: tokens {n} ppl mode2 {ppl['mode2']:.4f} i8 {ppl['i8']:.4f} "
          f"top1-agree {100 * agree:.2f}% mean|dlogp| {dnll:.5f}")
