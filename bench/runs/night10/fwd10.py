"""fwd10.py: night 10's copy of night 9's fwd.py. Changes: data in n10; sequences up to 8,192 tokens; an agent
episode is captured once, as its last turn, and every assistant span in it (each tool-call step and the final reply)
becomes training anchors (positions before a span are identical to that turn's own request, attention being causal);
activation peaks over every sequence except the KL hold-out (hash of the episode id % 20 in 0-2).
fwd.py: Artemis (Gemma 4 31B) forward pass one decoder layer at a time, weights streamed from disk, so the
whole model never has to fit on the 24 GB GPU. It writes what the MTP drafter reads at serving time, for every
position of every collected conversation:
  h      final_norm(x), the state that predicted the next token (BF16 [L, 5376])
  k58/v58, k59/v59   K and V of the last sliding (58) and last full (59) layer, as the drafter attends to them
  ti/tl  the target's top-32 next-token log-probs (softcap 30), the soft labels
It also measures every projection's input amax (the activation scale NVFP4 W4A4 prefill needs) and checks the
pass against the server's own greedy replies.
  python fwd.py tokenize                         -> n9/tok.pt
  python fwd.py run WEIGHTS DEVICE [MAX_SEQS]    WEIGHTS = bf16 | imw ; -> n9/states-WEIGHTS/*.pt, amax-WEIGHTS.json"""
import json, math, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open

D = "/opt/models/n10"
BF16 = "/opt/models/n8/A2/bf16"
IMW = "/opt/models/n9/A2-q-imw"
TEMPLATE = "/opt/models/templates/gemma4-agent.jinja"
PRE = "model.language_model."
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MAX_LEN = 8192
MODEL_OPEN, EMPTY_THOUGHT, TURN_END, TR_OPEN, TR_CLOSE = [105, 4368, 107], [100, 45518, 107, 101], 106, 50, 51


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kl_held(rid):
    import hashlib
    return int(hashlib.md5(rid.split("~")[0].encode()).hexdigest(), 16) % 20 in (0, 1, 2)


def spans(ids):
    """assistant-generated spans of a rendered sequence: after '<|turn>model\n' (and an empty thought block) or after a
    tool response the model continues from, up to and including the first '<turn|>' or '<|tool_response>'."""
    out, i, n = [], 0, len(ids)
    while i < n:
        st = None
        if ids[i:i + 3] == MODEL_OPEN:
            st = i + 3
            if ids[st:st + 4] == EMPTY_THOUGHT:
                st += 4
        elif ids[i] == TR_CLOSE and i + 1 < n and ids[i + 1] not in (TR_OPEN, TURN_END):
            st = i + 1
        if st is None:
            i += 1; continue
        j = st
        while j < n and ids[j] not in (TURN_END, TR_OPEN):
            j += 1
        out.append((st, min(j, n - 1)))
        i = j + 1
    return out


def tokenize():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BF16)
    tmpl = open(TEMPLATE).read()
    out, bad, mism, n = [], 0, 0, 0
    reasons = {}
    recs, last = [], {}
    for line in open(D + "/gen.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:
            reasons["cut-off line"] = reasons.get("cut-off line", 0) + 1; continue
        if r["cat"] == "agent":
            e = r["id"].split("~")[0]
            if e not in last or r["turn"] > last[e]["turn"]:
                last[e] = r
        else:
            recs.append(r)
    recs += [dict(r, id=e, turns=r["turn"] + 1) for e, r in last.items()]
    span_mismatch, anchors = 0, 0
    for r in recs:
        n += 1
        rep = r["reply"]
        am = {"role": "assistant", "content": rep.get("content") or ""}
        if rep.get("tool_calls"):
            calls = []
            for c in rep["tool_calls"]:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except Exception:
                    args = {}
                calls.append({"type": "function", "function": {"name": c["function"]["name"], "arguments": args}})
            am["tool_calls"] = calls
        kw = dict(tools=r.get("tools"), chat_template=tmpl, tokenize=False)
        try:
            p = tok.apply_chat_template(r["messages"], add_generation_prompt=True, **kw)
            f = tok.apply_chat_template(r["messages"] + [am], **kw)
        except Exception as e:
            reasons["template"] = reasons.get("template", 0) + 1; bad += 1; continue
        # the generation prompt ends with an empty thought block (thinking off) that a rendered past reply drops:
        # the sequence is the server's exact prompt plus the reply as the template renders it, up to <turn|>
        mark = "<|turn>model\n"
        tail = f[f.rfind(mark) + len(mark):]
        ends = [(tail.find(e), e) for e in ("<|tool_response>", "<turn|>") if tail.find(e) >= 0]
        if not ends:
            reasons["no end"] = reasons.get("no end", 0) + 1; bad += 1; continue
        end, e = min(ends)
        tail = tail[:end + len(e)]
        pi = tok(p, add_special_tokens=False).input_ids
        fi = tok(p + tail, add_special_tokens=False).input_ids
        if fi[:len(pi)] != pi:
            reasons["prefix"] = reasons.get("prefix", 0) + 1; bad += 1; continue
        u = r.get("usage") or {}
        if u.get("prompt_tokens") != len(pi):
            mism += 1
        r1 = len(fi)
        if r.get("finish") == "length" and u.get("completion_tokens"):
            r1 = min(r1, len(pi) + u["completion_tokens"])
        fi = fi[:min(r1, MAX_LEN)]
        if len(fi) - len(pi) < 2:
            reasons["short"] = reasons.get("short", 0) + 1; bad += 1; continue
        rec = {"id": r["id"], "cat": r["cat"], "temp": r["temperature"], "ids": fi, "r0": len(pi)}
        if r["cat"] == "agent":
            sp = [(a, b) for a, b in spans(fi) if a < len(fi)]
            if len(sp) != r["turns"]:
                span_mismatch += 1
            anc = [q - 1 for a, b in sp for q in range(a, min(b, len(fi) - 1) + 1)]
            if len(anc) < 2:
                reasons["no spans"] = reasons.get("no spans", 0) + 1; bad += 1; continue
            rec["anc"] = anc; rec["r0"] = sp[0][0]
            anchors += len(anc)
        else:
            anchors += len(fi) - len(pi)
        out.append(rec)
    torch.save(out, D + "/tok.pt")
    toks = sum(len(x["ids"]) for x in out); resp = sum(len(x["ids"]) - x["r0"] for x in out)
    log(f"TOKENIZED {len(out)} of {n} (bad {bad}, prompt count differs from the server {mism}, agent span count differs {span_mismatch}, reasons {reasons}); {toks} tokens, {anchors} anchors")


class Weights:
    def __init__(self, kind, dev="cpu"):
        self.kind, self.dev = kind, dev
        root = BF16 if kind == "bf16" else IMW
        idx = json.load(open(root + "/model.safetensors.index.json"))["weight_map"]
        self.root, self.idx, self.handles = root, idx, {}
        self.bf = json.load(open(BF16 + "/model.safetensors.index.json"))["weight_map"]

    def _get(self, root, idx, name):
        f = idx[name]
        key = (root, f)
        if key not in self.handles:
            self.handles[key] = safe_open(os.path.join(root, f), "pt")
        return self.handles[key].get_tensor(name)

    def get(self, name):
        """BF16 tensor; an NVFP4 projection (imw) comes back dequantized."""
        if self.kind == "imw" and name in self.idx:
            t = self._get(self.root, self.idx, name)
            if t.dtype == torch.uint8:
                t = t.to(self.dev)
                bs = self._get(self.root, self.idx, name + "_scale").to(self.dev).float()
                s2 = self._get(self.root, self.idx, name + "_scale_2").to(self.dev).float()
                rows = t.shape[0]
                codes = torch.stack([t & 15, t >> 4], -1).view(rows, -1)
                vals = E2M1.to(self.dev)[(codes & 7).long()] * torch.where(codes & 8 > 0, -1.0, 1.0)
                return (vals.view(rows, -1, 16) * bs.unsqueeze(-1) * s2).view(rows, -1).to(torch.bfloat16)
            return t.to(torch.bfloat16) if t.is_floating_point() else t
        return self._get(BF16, self.bf, name)


def text_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    c = json.load(open(BF16 + "/config.json"))["text_config"]
    cfg = Gemma4TextConfig(**c)
    cfg._attn_implementation = "sdpa"
    return cfg


def load_layer(cfg, i, W, dev):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(cfg, i)
    sd = {}
    for name in layer.state_dict().keys():
        sd[name] = W.get(PRE + f"layers.{i}." + name)
    layer.to_empty(device=dev)
    layer.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)
    return layer.to(torch.bfloat16).eval()


def masks(lens, L, window, dev):
    q = torch.arange(L, device=dev)
    causal = q[None, :] <= q[:, None]                       # [Lq, Lk]
    sliding = causal & (q[None, :] > q[:, None] - window)
    keyok = torch.arange(L, device=dev)[None, :] < torch.tensor(lens, device=dev)[:, None]  # [B, Lk]
    full = (causal[None] & keyok[:, None, :]) | torch.eye(L, dtype=torch.bool, device=dev)[None]
    slid = (sliding[None] & keyok[:, None, :]) | torch.eye(L, dtype=torch.bool, device=dev)[None]
    return full[:, None], slid[:, None]


def microbatches(seqs, tok_budget=32768, attn_budget=1 << 24):
    seqs = sorted(seqs, key=lambda s: len(s["ids"]))
    out, cur = [], []
    for s in seqs:
        L = len(s["ids"])
        if cur and ((len(cur) + 1) * L > tok_budget or (len(cur) + 1) * L * L > attn_budget):
            out.append(cur); cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def run(kind, dev, max_seqs=None, pass_tokens=360_000, calib_tokens=10 ** 12):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding, Gemma4RMSNorm
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = text_config()
    W = Weights(kind, dev)
    data = torch.load(D + "/tok.pt")
    if os.environ.get("FWD_TEST"):
        data = sorted([s for s in data if s["temp"] == 0.0], key=lambda s: len(s["ids"]))[:int(max_seqs or 2)]
    elif max_seqs:
        data = data[:int(max_seqs)]
    outdir = f"{D}/states-{kind}" + ("-test" if os.environ.get("FWD_TEST") else "")
    os.makedirs(outdir, exist_ok=True)
    rot = Gemma4TextRotaryEmbedding(cfg, device=dev)
    passes, cur, n = [], [], 0
    for s in data:
        cur.append(s); n += len(s["ids"])
        if n >= pass_tokens:
            passes.append(cur); cur, n = [], 0
    if cur:
        passes.append(cur)
    amax, calib_seen = {}, 0
    agree = [0, 0]
    shard = 0
    t0 = time.time()
    for pi, seqs in enumerate(passes):
        mbs = microbatches(seqs)
        log(f"pass {pi + 1}/{len(passes)}: {len(seqs)} sequences, {sum(len(s['ids']) for s in seqs)} tokens, {len(mbs)} micro-batches")
        emb = W.get(PRE + "embed_tokens.weight").to(dev)
        scale = torch.tensor(cfg.hidden_size ** 0.5, dtype=torch.bfloat16, device=dev)
        H, META = [], []
        for mb in mbs:
            L = max(len(s["ids"]) for s in mb)
            ids = torch.zeros(len(mb), L, dtype=torch.long)
            for b, s in enumerate(mb):
                ids[b, :len(s["ids"])] = torch.tensor(s["ids"])
            H.append(F.embedding(ids.to(dev), emb) * scale)
            META.append((mb, L, [len(s["ids"]) for s in mb]))
        del emb
        calib_mbs = set()
        if calib_seen < calib_tokens:
            for m, (mb, L, lens) in enumerate(META):
                if calib_seen >= calib_tokens:
                    break
                calib_mbs.add(m); calib_seen += sum(lens)
        KV = [dict() for _ in META]
        for i in range(cfg.num_hidden_layers):
            lt = cfg.layer_types[i]
            layer = load_layer(cfg, i, W, dev)
            state = {"valid": None}
            hooks = []
            for nm, mod in (("qkv", layer.self_attn.q_proj), ("o", layer.self_attn.o_proj),
                            ("gate_up", layer.mlp.gate_proj), ("down", layer.mlp.down_proj)):
                def pre(module, args, nm=nm):
                    if state["valid"] is not None:
                        x = args[0]
                        a = (x.float().abs().amax(-1) * state["valid"]).max().item()
                        key = f"{i}.{nm}"
                        amax[key] = max(amax.get(key, 0.0), a)
                hooks.append(mod.register_forward_pre_hook(pre))
            for m, (mb, L, lens) in enumerate(META):
                pos = torch.arange(L, device=dev)[None].expand(len(mb), -1)
                pe = rot(H[m], pos, lt)
                full, slid = masks(lens, L, cfg.sliding_window, dev)
                keep = torch.tensor([0.0 if kl_held(s["id"]) else 1.0 for s in mb], device=dev)[:, None]
                state["valid"] = (torch.arange(L, device=dev)[None] < torch.tensor(lens, device=dev)[:, None]).float() * keep if m in calib_mbs else None
                skv = {}
                H[m] = layer(H[m], None, shared_kv_states=skv, position_embeddings=pe,
                             attention_mask=slid if lt == "sliding_attention" else full, position_ids=pos)
                if i in (58, 59):
                    k, v = skv[lt]
                    tag = "58" if i == 58 else "59"
                    KV[m]["k" + tag] = k.to("cpu", non_blocking=True)
                    KV[m]["v" + tag] = v.to("cpu", non_blocking=True)
            for h in hooks:
                h.remove()
            del layer
            if dev == "cuda":
                torch.cuda.empty_cache()
            if i % 10 == 9:
                log(f"  layer {i + 1}/60 ({time.time() - t0:.0f}s)")
        norm = Gemma4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev)
        norm.load_state_dict({"weight": W.get(PRE + "norm.weight").to(dev)})
        norm = norm.to(torch.bfloat16)
        emb = W.get(PRE + "embed_tokens.weight").to(dev)
        cap = cfg.final_logit_softcapping
        for m, (mb, L, lens) in enumerate(META):
            hn = norm(H[m])
            ti = torch.empty(len(mb), L, 32, dtype=torch.int32, device=dev)
            tl = torch.empty(len(mb), L, 32, dtype=torch.float16, device=dev)
            flat = hn.view(-1, hn.shape[-1])
            for a in range(0, flat.shape[0], 1024):
                z = (flat[a:a + 1024] @ emb.T).float()
                if cap:
                    z = torch.tanh(z / cap) * cap
                lp = torch.log_softmax(z, -1)
                v, ix = lp.topk(32, -1)
                ti.view(-1, 32)[a:a + 1024] = ix.int(); tl.view(-1, 32)[a:a + 1024] = v.half()
            recs = []
            for b, s in enumerate(mb):
                n = lens[b]
                if s["temp"] == 0.0:
                    pos = torch.tensor(s["anc"]) if "anc" in s else torch.arange(s["r0"] - 1, n - 1)
                    pred = ti[b, pos, 0].cpu()
                    want = torch.tensor(s["ids"], dtype=torch.int32)[pos + 1]
                    agree[0] += int((pred == want).sum()); agree[1] += len(want)
                recs.append({"id": s["id"], "cat": s["cat"], "temp": s["temp"], "ids": torch.tensor(s["ids"], dtype=torch.int32),
                             **({"anc": torch.tensor(s["anc"], dtype=torch.int32)} if "anc" in s else {}),
                             "r0": s["r0"], "h": hn[b, :n].cpu(), "ti": ti[b, :n].cpu(), "tl": tl[b, :n].cpu(),
                             **{k: t[b, :, :n].contiguous() for k, t in KV[m].items()}})
            torch.save(recs, f"{outdir}/shard{shard:04d}.pt"); shard += 1
        del emb, H, KV
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"pass {pi + 1} written; greedy agreement so far {agree[0]}/{agree[1]} = {100 * agree[0] / max(1, agree[1]):.2f}%")
    json.dump({"amax": amax, "calib_tokens": calib_seen, "greedy_agreement": [agree[0], agree[1]]},
              open(f"{D}/amax-{kind}.json", "w"), indent=0)
    log(f"FWD_DONE {kind}: {shard} shards, greedy agreement {agree[0]}/{agree[1]} = {100 * agree[0] / max(1, agree[1]):.2f}%, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] == "tokenize":
        tokenize()
    else:
        run(sys.argv[2], sys.argv[3], *(sys.argv[4:5]))
