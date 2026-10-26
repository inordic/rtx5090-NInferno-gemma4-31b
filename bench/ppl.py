#!/usr/bin/env python3
"""Perplexity of `ninferno-gemma4 score --dump` files (FP32 log-probs, then I32 argmax ids)."""
import math, os, struct, sys
for f in sys.argv[1:]:
    raw = open(f, "rb").read()
    n = len(raw) // 8
    lp = struct.unpack(f"<{n}f", raw[:4 * n])
    name = os.path.basename(f)
    print(f"PPL {name}: tokens {n} ppl {math.exp(-sum(lp) / n):.4f}")
