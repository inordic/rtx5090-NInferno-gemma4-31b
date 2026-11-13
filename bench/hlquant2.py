"""hlquant2: hlquant's NVFP4 layout with a per-block scale search. One pass writes up to three checkpoints:
  rtn  the plain amax/6 block scale (hlquant's output)
  mse  per 16-weight block, the E4M3 scale among candidates that minimises sum (w - q)^2
  imw  the same, weighted by the importance matrix: sum h_j (w_j - q_j)^2, h_j = mean x_j^2 of that
       matrix's input channel j (llama.cpp imatrix: in_sum2 / counts), the diagonal of the local Hessian
Same E2M1 codes, per-16 E4M3 block scales, per-tensor scale_2 and LilaRest's input scales, so
ninferno-convert's gemma4_31b_nvfp4 recipe takes every output unchanged.

  python hlquant2.py SRC_DIR DST_PREFIX SCALES.json [IMATRIX.gguf] [--offsets -4,-3,-2,-1,1,2,5]
  -> DST_PREFIX-rtn, DST_PREFIX-mse, DST_PREFIX-imw (imw only with an imatrix)"""
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/root/gbench")
from hlquant import E2M1_BOUNDS, E2M1_VALUES, PROJ  # noqa: E402

GGUF_NAME = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output",
             "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}
INPUT_PEER = {"attn_k": "attn_q", "attn_v": "attn_q", "ffn_up": "ffn_gate"}  # matrices that read the same input


def load_imatrix(path):
    sys.path.insert(0, "/opt/llama.cpp/gguf-py")
    from gguf import GGUFReader
    r = GGUFReader(path)
    sums, counts = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[: -len(".in_sum2")]] = torch.tensor(t.data.copy()).float().flatten()
        elif t.name.endswith(".counts"):
            counts[t.name[: -len(".counts")]] = float(t.data.flatten()[0])
    return {k: v / max(counts.get(k, 1.0), 1.0) for k, v in sums.items()}


def importance(imx, stem):
    m = PROJ.match(stem + ".weight")
    layer, kind = m.group(1), m.group(3) or m.group(4)
    g = GGUF_NAME[kind]
    for name in (f"blk.{layer}.{g}.weight", f"blk.{layer}.{INPUT_PEER.get(g, g)}.weight"):
        if name in imx:
            return imx[name]
    return None


# quarter-step lookup: |x| * 4 floored -> E2M1 code (thresholds 0.25 0.75 1.25 1.75 2.5 3.5 5 are all
# multiples of 0.25); a value exactly on a threshold goes up, which hlquant's searchsorted differs on only
# for exact ties
LUT = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7], dtype=torch.uint8)


def e2m1(scaled):
    """E2M1 codes (0..7) and signs of scaled values, round to nearest, saturating at 6"""
    sign = scaled < 0
    idx = scaled.abs().mul_(4).clamp_(max=24).to(torch.uint8)
    return LUT.to(scaled.device)[idx.long()], sign


def pack(code, sign, rows, k):
    c = (code.to(torch.uint8) | (sign.to(torch.uint8) << 3)).view(rows, k)
    return ((c[:, 1::2] << 4) | c[:, 0::2]).contiguous()


def search(w, s2, h, offsets, chunk=256, methods=("rtn", "mse", "imw")):
    """-> {method: (packed U8, E4M3 block scales)} and the summed objectives per method"""
    w = w.float()
    rows, k = w.shape
    nb = k // 16
    hb = h.view(nb, 16) if h is not None else None
    out = {m: ([], []) for m in methods if m != "imw" or h is not None}
    gain = {"mse": [0.0, 0.0], "imw": [0.0, 0.0]}  # [rtn objective, chosen objective]
    for r0 in range(0, rows, chunk):
        blocks = w[r0:r0 + chunk].view(-1, nb, 16)
        n = blocks.shape[0]
        base = blocks.abs().amax(-1) / (6.0 * s2)
        base[base == 0] = 1.0
        base_code = base.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
        best = {m: (torch.full((n, nb), float("inf"), device=w.device), base_code.clone()) for m in ("mse", "imw") if m in out}
        rtn_obj = {}
        for off in [0] + list(offsets):
            cc = (base_code + off).clamp(1, 126)
            sc = cc.to(torch.uint8).view(torch.float8_e4m3fn).float()
            scaled = blocks / (sc.unsqueeze(-1) * s2)
            code, sign = e2m1(scaled)
            q = E2M1_VALUES.to(blocks.device)[code.long()] * torch.where(sign, -1.0, 1.0) * sc.unsqueeze(-1) * s2
            err = (blocks - q).pow_(2)
            objs = {"mse": err.sum(-1)} if "mse" in out else {}
            if "imw" in out:
                objs["imw"] = (err * hb).sum(-1)
            for m, o in objs.items():
                if off == 0:
                    rtn_obj[m] = o.sum().item()
                bo, bc = best[m]
                better = o < bo
                bo[better] = o[better]
                bc[better] = cc[better]
            if off == 0 and "rtn" in out:
                out["rtn"][0].append(pack(code, sign, n, k))
                out["rtn"][1].append(base_code.to(torch.uint8).view(torch.float8_e4m3fn))
        for m, (bo, bc) in best.items():
            sc = bc.to(torch.uint8).view(torch.float8_e4m3fn)
            code, sign = e2m1(blocks / (sc.float().unsqueeze(-1) * s2))
            out[m][0].append(pack(code, sign, n, k))
            out[m][1].append(sc)
            gain[m][0] += rtn_obj[m]
            gain[m][1] += bo.sum().item()
    return {m: (torch.cat(p).cpu(), torch.cat(s).cpu()) for m, (p, s) in out.items()}, gain


class Writer:
    def __init__(self, dst):
        self.dst, self.shard, self.bytes, self.no, self.map = dst, {}, 0, 1, {}
        os.makedirs(dst, exist_ok=True)

    def add(self, k, t):
        self.shard[k] = t.contiguous()
        self.bytes += t.numel() * t.element_size()
        if self.bytes > 5 * 2**30:
            self.flush()

    def flush(self):
        if not self.shard:
            return
        f = f"model-{self.no:05d}.safetensors"
        save_file(self.shard, os.path.join(self.dst, f), metadata={"format": "pt"})
        self.map.update({k: f for k in self.shard})
        self.shard, self.bytes, self.no = {}, 0, self.no + 1

    def close(self):
        self.flush()
        json.dump({"metadata": {}, "weight_map": self.map},
                  open(os.path.join(self.dst, "model.safetensors.index.json"), "w"), indent=1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    offsets = [-4, -3, -2, -1, 1, 2, 5]
    dev, methods_arg, blend = "cpu", None, None
    for a in sys.argv[1:]:
        if a.startswith("--offsets="):
            offsets = [int(x) for x in a.split("=", 1)[1].split(",")]
        if a.startswith("--device="):
            dev = a.split("=", 1)[1]
        if a.startswith("--blend="):
            blend = float(a.split("=", 1)[1])
        if a.startswith("--methods="):
            methods_arg = a.split("=", 1)[1].split(",")
    src, prefix, scales_path = args[:3]
    imx = load_imatrix(args[3]) if len(args) > 3 else None
    if imx is not None:
        print(f"imatrix: {len(imx)} matrices", flush=True)
    scales = json.load(open(scales_path))
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    handles = {f: safe_open(os.path.join(src, f), "pt") for f in set(index.values())}
    get = lambda n: handles[index[n]].get_tensor(n)
    names = sorted(n for n in index if n.startswith("model.language_model."))
    groups = {}
    for n in names:
        m = PROJ.match(n)
        if m:
            fam = "qkv" if m.group(3) in ("q", "k", "v") else ("gu" if m.group(4) in ("gate", "up") else n)
            groups.setdefault((m.group(1), fam), []).append(n[: -len(".weight")])
    methods = methods_arg or (["rtn", "mse"] + (["imw"] if imx is not None else []))
    chunk = 4096 if dev != "cpu" else 256
    writers = {m: Writer(f"{prefix}-{m}") for m in methods}
    tot = {"mse": [0.0, 0.0], "imw": [0.0, 0.0]}
    missing, done, t0 = 0, set(), time.time()
    for (layer, fam), stems in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        ws = {s: get(s + ".weight") for s in stems}
        s2 = max(float(w.float().abs().max()) for w in ws.values()) / (6.0 * 448.0)
        act = max(scales[s + ".input_scale"] for s in stems)
        for s, w in ws.items():
            h = importance(imx, s) if imx is not None else None
            if imx is not None and (h is None or h.numel() != w.shape[1]):
                missing += 1
                h = torch.ones(w.shape[1])
            if h is not None and blend is not None:
                h = h / h.mean() + blend  # importance mixed with uniform weight
            res, gain = search(w.to(dev), s2, h.to(dev) if h is not None else None, offsets, chunk, methods)
            for m in tot:
                tot[m][0] += gain[m][0]
                tot[m][1] += gain[m][1]
            for m in methods:
                p, sc = res[m]
                wr = writers[m]
                wr.add(s + ".weight", p)
                wr.add(s + ".weight_scale", sc)
                wr.add(s + ".weight_scale_2", torch.tensor(s2, dtype=torch.float32))
                wr.add(s + ".input_scale", torch.tensor(act, dtype=torch.float32))
            done.add(s + ".weight")
        msg = " ".join(f"{m} {1 - tot[m][1] / tot[m][0]:.1%}" for m in tot if tot[m][0] > 0)
        print(f"layer {layer} {fam}: {len(stems)} matrices, objective cut so far: {msg}, {time.time() - t0:.0f}s",
              flush=True)
    for n in names:
        if n not in done:
            t = get(n)
            for wr in writers.values():
                wr.add(n, t)
    for wr in writers.values():
        wr.close()
    print(f"imatrix missing for {missing} matrices" if imx is not None else "no imatrix")
    print(f"QUANTIZE2 DONE: {', '.join(methods)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
