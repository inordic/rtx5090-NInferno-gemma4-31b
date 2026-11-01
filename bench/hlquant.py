"""hlquant: a ModelOpt-style NVFP4 checkpoint of a BF16 Gemma 4 31B fine-tune, with the activation
divisors of LilaRest/gemma-4-31B-it-NVFP4-turbo (ModelOpt 0.37.0, NVFP4, max calibration), so that
ninferno-convert's gemma4_31b_nvfp4 recipe takes it as it takes LilaRest's and allows W4A4 prefill.

Weights are quantized as ModelOpt's NVFP4 does (per-tensor scale_2 = amax / (6 * 448); per-16 E4M3
block scales; E2M1 codes rounded to nearest, ties to even; low nibble first). Matrices one fused
parent will hold (q/k/v; gate/up) share scale_2 and input_scale (the larger of LilaRest's), so the
converter packs them without rounding again.

  python hlquant.py fetch-scales OUT.json                  LilaRest's input_scale / weight_scale_2
  python hlquant.py selftest REF_BF16_REPO NAME            quantize one BF16 tensor of Google's model
                                                           and compare with LilaRest's codes/scales
  python hlquant.py quantize SRC_DIR DST_DIR SCALES.json   the whole checkpoint (text tensors only)

Run with /opt/llama.cpp/.venv/bin/python (torch CPU, safetensors)."""
import json
import os
import re
import struct
import sys
import time
import urllib.request

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LILA = "LilaRest/gemma-4-31B-it-NVFP4-turbo"
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
PROJ = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def http(url, rng=None):
    headers = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def remote_index(repo):
    """{tensor name: (url, dtype, shape, absolute byte range)} from the shards' headers."""
    base = f"https://huggingface.co/{repo}/resolve/main/"
    files = [f["path"] for f in json.loads(http(f"https://huggingface.co/api/models/{repo}/tree/main"))
             if f["path"].endswith(".safetensors")]
    out = {}
    for f in files:
        n = struct.unpack("<Q", http(base + f, (0, 7)))[0]
        header = json.loads(http(base + f, (8, 8 + n - 1)))
        for name, info in header.items():
            if name != "__metadata__":
                a, b = info["data_offsets"]
                out[name] = (base + f, info["dtype"], info["shape"], (8 + n + a, 8 + n + b - 1))
    return out


DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn}


def remote_tensor(index, name):
    url, dtype, shape, rng = index[name]
    raw = bytearray(http(url, rng))
    return torch.frombuffer(raw, dtype=DTYPES[dtype]).reshape(shape) if shape else \
        torch.frombuffer(raw, dtype=DTYPES[dtype])[0]


def quantize(w, scale_2):
    """ModelOpt NVFP4 of a [rows, k] matrix against a given scale_2: (packed U8, E4M3 block scales)."""
    w = w.float()
    rows, k = w.shape
    blocks = w.view(rows, k // 16, 16)
    block_scale = blocks.abs().amax(-1) / (6.0 * scale_2)
    block_scale[block_scale == 0] = 1.0
    block_scale = block_scale.to(torch.float8_e4m3fn)
    scaled = blocks / (block_scale.float().unsqueeze(-1) * scale_2)
    sign = (scaled < 0).to(torch.uint8)
    mag = scaled.abs()
    code = torch.searchsorted(E2M1_BOUNDS, mag.contiguous(), out_int32=True)
    # ties at 0.75, 1.75, 3.5 go to the even code below (searchsorted puts them above)
    for b in (0.75, 1.75, 3.5):
        code -= (mag == b).to(torch.int32)
    code = (code.to(torch.uint8) | (sign << 3)).view(rows, k)
    packed = (code[:, 1::2] << 4) | code[:, 0::2]
    return packed.contiguous(), block_scale.contiguous()


def dequantize(packed, block_scale, scale_2):
    rows = packed.shape[0]
    lo, hi = packed & 15, packed >> 4
    codes = torch.stack([lo, hi], -1).view(rows, -1)
    vals = E2M1_VALUES[(codes & 7).long()] * torch.where(codes & 8 > 0, -1.0, 1.0)
    return (vals.view(rows, -1, 16) * block_scale.float().unsqueeze(-1) * scale_2).view(rows, -1)


def cmd_fetch_scales(out):
    index = remote_index(LILA)
    names = sorted(n for n in index if n.endswith(".input_scale") or n.endswith(".weight_scale_2"))
    scales = {}
    for i, n in enumerate(names):
        scales[n] = float(remote_tensor(index, n))
        if i % 100 == 0:
            print(f"{i}/{len(names)} {n} {scales[n]:.6g}", flush=True)
    json.dump(scales, open(out, "w"), indent=0)
    print(f"wrote {len(scales)} scalars to {out}")


def cmd_selftest(ref_repo, name):
    lila, ref = remote_index(LILA), remote_index(ref_repo)
    stem = name[: -len(".weight")]
    w = remote_tensor(ref, name)
    packed_l = remote_tensor(lila, name)
    scale_l = remote_tensor(lila, stem + ".weight_scale")
    s2_l = float(remote_tensor(lila, stem + ".weight_scale_2"))
    s2 = float(w.float().abs().max()) / (6.0 * 448.0)
    print(f"{name}: shape {list(w.shape)}, scale_2 ours {s2:.8g} LilaRest {s2_l:.8g}")
    packed, scale = quantize(w, s2_l)
    same_codes = (packed == packed_l).float().mean().item()
    same_scales = (scale.view(torch.uint8) == scale_l.view(torch.uint8)).float().mean().item()
    err = (dequantize(packed, scale, s2_l) - w.float()).norm() / w.float().norm()
    err_l = (dequantize(packed_l, scale_l, s2_l) - w.float()).norm() / w.float().norm()
    print(f"against LilaRest's scale_2: code bytes identical {same_codes:.6%}, block scales identical "
          f"{same_scales:.6%}; relative error ours {err:.5f}, LilaRest {err_l:.5f}")


def cmd_quantize(src, dst, scales_path):
    scales = json.load(open(scales_path))
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    handles = {f: safe_open(os.path.join(src, f), "pt") for f in set(index.values())}
    get = lambda n: handles[index[n]].get_tensor(n)
    names = sorted(n for n in index if n.startswith("model.language_model."))
    groups = {}  # (layer, family) -> member stems sharing scale_2 and input_scale
    for n in names:
        m = PROJ.match(n)
        if m:
            fam = "qkv" if m.group(3) in ("q", "k", "v") else ("gu" if m.group(4) in ("gate", "up") else n)
            groups.setdefault((m.group(1), fam), []).append(n[: -len(".weight")])
    shard, shard_bytes, shard_no, weight_map, t0 = {}, 0, 1, {}, time.time()

    def flush(final=False):
        nonlocal shard, shard_bytes, shard_no
        if not shard:
            return
        fname = f"model-{shard_no:05d}.safetensors"
        save_file(shard, os.path.join(dst, fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
        print(f"wrote {fname}: {len(shard)} tensors, {shard_bytes / 2**30:.2f} GiB, {time.time() - t0:.0f}s", flush=True)
        shard, shard_bytes, shard_no = {}, 0, shard_no + 1

    def add(k, t):
        nonlocal shard_bytes
        shard[k] = t.contiguous()
        shard_bytes += t.numel() * t.element_size()
        if shard_bytes > 5 * 2**30:
            flush()

    done = set()
    for (layer, fam), stems in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        ws = {s: get(s + ".weight") for s in stems}
        s2 = max(float(w.float().abs().max()) for w in ws.values()) / (6.0 * 448.0)
        act = max(scales[s + ".input_scale"] for s in stems)
        for s, w in ws.items():
            packed, block_scale = quantize(w, s2)
            add(s + ".weight", packed)
            add(s + ".weight_scale", block_scale)
            add(s + ".weight_scale_2", torch.tensor(s2, dtype=torch.float32))
            add(s + ".input_scale", torch.tensor(act, dtype=torch.float32))
            done.add(s + ".weight")
        print(f"layer {layer} {fam}: {len(stems)} matrices, scale_2 {s2:.4g} (LilaRest "
              f"{max(scales[s + '.weight_scale_2'] for s in stems):.4g}), input_scale {act:.4g}", flush=True)
    for n in names:
        if n not in done:
            add(n, get(n))
    flush()
    json.dump({"metadata": {}, "weight_map": weight_map},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
    print(f"QUANTIZE DONE: {len(weight_map)} tensors in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"fetch-scales": cmd_fetch_scales, "selftest": cmd_selftest, "quantize": cmd_quantize}[cmd](*sys.argv[2:])
