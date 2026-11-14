"""calib10.py SRC OUT_V1 OUT_V2 AMAX_JSON... : NVFP4 activation scales calibrated for Artemis.
SRC is production's quantized checkpoint (importance-weighted NVFP4 weights with LilaRest's activation scales; it
converts to production's artifact byte for byte, apart from the artifact id). Only the activation scales change; the
weights stay bit-identical. The convention is NVFP4's: input_scale = amax / (448 * 6).
  V1: every projection whose measured peak exceeds its scale's range gets the peak plus 25% headroom; no scale is lowered.
  V2: halfway (geometric mean) between the old scale and V1's.
Peaks come from the AMAX_JSON files of fwd.py / fwd10.py (keys 'LAYER.qkv|o|gate_up|down'), merged by maximum.
Writes OUT_V1 and OUT_V2 (shards without a changed tensor are hard-linked) and calib.json next to OUT_V1."""
import json, math, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

src, out1, out2, amax_files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
RANGE = 448.0 * 6.0
HEAD = 1.25
PRE = "model.language_model.layers."
GROUPS = {"qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o": ["self_attn.o_proj"],
          "gate_up": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"]}

amax = {}
for f in amax_files:
    for k, v in json.load(open(f))["amax"].items():
        amax[k] = max(amax.get(k, 0.0), v)
idx = json.load(open(src + "/model.safetensors.index.json"))["weight_map"]
cur = {}
handles = {}
for name, shard in idx.items():
    if name.startswith(PRE) and name.endswith(".input_scale"):
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(src, shard), "pt")
        cur[name] = handles[shard].get_tensor(name)

new1, new2, report = {}, {}, []
for key, a in sorted(amax.items(), key=lambda kv: (int(kv[0].split(".")[0]), kv[0])):
    layer, grp = key.split(".", 1)
    names = [f"{PRE}{layer}.{p}.input_scale" for p in GROUPS[grp] if f"{PRE}{layer}.{p}.input_scale" in cur]
    if not names:
        continue
    s0 = max(float(cur[n].float().max()) for n in names)
    want = HEAD * a / RANGE
    s1 = max(s0, want)
    s2 = max(s0, math.sqrt(s0 * s1))
    report.append({"key": key, "peak": round(a, 3), "range_before": round(s0 * RANGE, 3), "clips_before": a > s0 * RANGE,
                   "ratio": round(a / (s0 * RANGE), 3), "scale_before": s0, "scale_v1": s1, "scale_v2": s2})
    if s1 > s0:
        for n in names:
            new1[n] = torch.full_like(cur[n], s1)
            new2[n] = torch.full_like(cur[n], s2)


def write(out, new):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if f.endswith(".safetensors"):
            names = [n for n, s in idx.items() if s == f]
            if not any(n in new for n in names):
                if not os.path.exists(os.path.join(out, f)):
                    os.link(p, os.path.join(out, f))
                continue
            with safe_open(p, "pt") as h:
                meta = h.metadata()
            t = load_file(p)
            for n in names:
                if n in new:
                    assert t[n].shape == new[n].shape and t[n].dtype == new[n].dtype
                    t[n] = new[n]
            save_file(t, os.path.join(out, f), metadata=meta)
            del t
        elif os.path.isfile(p):
            shutil.copy(p, os.path.join(out, f))
    print("wrote", out, len(new), "scales changed", flush=True)


clipped = [r for r in report if r["clips_before"]]
print(f"{len(report)} projection groups with peaks; {len(clipped)} clip under the old scales "
      f"(worst {max((r['ratio'] for r in report), default=0):.2f}x); V1 raises {len(new1)} scales", flush=True)
json.dump({"head": HEAD, "range": RANGE, "amax_files": amax_files, "groups": report},
          open(os.path.join(os.path.dirname(out1.rstrip("/")), "calib.json"), "w"), indent=0)
if os.environ.get("DRY"):
    sys.exit(0)
write(out1, new1)
write(out2, new2)
print("CALIB_DONE", flush=True)
