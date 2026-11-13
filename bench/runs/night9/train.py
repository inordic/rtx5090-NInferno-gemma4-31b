"""train.py: fine-tune Google's Gemma 4 31B MTP assistant (the drafter) on Artemis's own states.
Each training anchor t is a reply position. Step 1 feeds [target embedding of x(t+1); final_norm h(t)], queries at
position t+1 and attends to the target's layer-58/59 K/V up to t, exactly as NInferno's MtpRound does; step k feeds
the true token x(t+k) and the drafter's own carried state. Each step k learns the target's top-32 distribution at
position t+k (soft cross-entropy). Validation reports per-step top-1 agreement and the simulated accepted length of
a 7-token draft that feeds its own tokens, for the base drafter and every epoch.
  python train.py eval  STATES_DIR                     -> the base drafter's numbers
  python train.py train STATES_DIR OUT_DIR [EPOCHS] [LR]"""
import glob, hashlib, json, math, os, random, shutil, sys, time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file

BASE = "/opt/models/dl2/assistant"
BF16 = "/opt/models/n8/A2/bf16"
K = 7
STEP_W = [0.85 ** k for k in range(K)]
dev = os.environ.get("DEV", "cuda")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def is_val(rid):
    return int(hashlib.md5(rid.encode()).hexdigest(), 16) % 20 == 0


def load_assistant():
    from transformers import Gemma4AssistantForCausalLM
    m = Gemma4AssistantForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    m.config.get_text_config()._attn_implementation = "sdpa"
    m.model.config._attn_implementation = "sdpa"
    return m.to(dev)


def target_embedding():
    from safetensors import safe_open
    idx = json.load(open(BF16 + "/model.safetensors.index.json"))["weight_map"]
    n = "model.language_model.embed_tokens.weight"
    with safe_open(os.path.join(BF16, idx[n]), "pt") as f:
        w = f.get_tensor(n).to(dev)
    return w, torch.tensor(5376 ** 0.5, dtype=torch.bfloat16, device=dev)


def records(states, val):
    files = sorted(glob.glob(states + "/shard*.pt"))
    return files


def iterate(files, val, shuffle, seed=0):
    rnd = random.Random(seed)
    files = list(files)
    if shuffle:
        rnd.shuffle(files)
    for f in files:
        recs = [r for r in torch.load(f) if is_val(r["id"]) == val or os.environ.get("TRAIN_TEST")]
        if shuffle:
            rnd.shuffle(recs)
        for r in recs:
            yield r


def masks(t, L, window):
    """queries are anchors t (absolute positions), keys 0..L-1: full sees k <= t, sliding sees t-window < k <= t."""
    k = torch.arange(L, device=dev)[None, :]
    tt = t[:, None]
    full = k <= tt
    slid = full & (k > tt - window)
    return full[None, None], slid[None, None]


def soft_ce(last, head_w, ti, tl):
    with torch.autocast(dev, dtype=torch.bfloat16):
        z = last @ head_w.T
    z = z.float()
    lse = torch.logsumexp(z, -1, keepdim=True)
    lq = z.gather(-1, ti.long()) - lse
    p = torch.softmax(tl.float(), -1)
    return -(p * lq).sum(-1), z.argmax(-1)


def unroll(m, E, scale, r, train, own_tokens=False, a0=0, a1=None):
    """One sequence, anchors a0..a1 of its reply. Returns per-step losses and argmax drafts for those anchors."""
    L = r["ids"].numel()
    r0 = r["r0"]
    t = torch.arange(r0 - 1, L - 1, device=dev)[a0:a1]   # anchors
    ids = r["ids"].to(dev).long()
    h = r["h"].to(dev).to(torch.bfloat16)
    ti, tl = r["ti"].to(dev), r["tl"].to(dev)
    skv = {"sliding_attention": (r["k58"][None].to(dev).to(torch.bfloat16), r["v58"][None].to(dev).to(torch.bfloat16)),
           "full_attention": (r["k59"][None].to(dev).to(torch.bfloat16), r["v59"][None].to(dev).to(torch.bfloat16))}
    full, slid = masks(t, L, 1024)
    qpos = (t + 1)[None]
    carry = h[t][None]                                     # [1, Q, 5376]
    tok = ids[t + 1]
    losses, drafts, valid = [], [], []
    head_w = m.lm_head.weight
    for k in range(K):
        lab = t + 1 + k                                    # position whose next-token distribution step k learns
        ok = lab <= L - 1
        if not ok.any():
            break
        with torch.autocast(dev, dtype=torch.bfloat16):
            emb = (F.embedding(tok, E) * scale)[None]
            x = m.pre_projection(torch.cat([emb, carry.to(emb.dtype)], -1))
            body = lambda x_: m.model(inputs_embeds=x_, attention_mask={"full_attention": full, "sliding_attention": slid},
                                      position_ids=qpos, shared_kv_states=skv, use_cache=False).last_hidden_state
            hs = checkpoint(body, x, use_reentrant=False) if train else body(x)   # train: recomputed per step in backward
            last = hs[0]
            carry = m.post_projection(hs)
        labc = lab.clamp(max=L - 1)
        li, lv = ti[labc], tl[labc]
        Q = last.shape[0]
        lk = torch.empty(Q, device=dev); dk = torch.empty(Q, dtype=torch.long, device=dev)
        parts_l, parts_d = [], []
        for a in range(0, Q, 1024):
            if train:
                l_, d_ = checkpoint(soft_ce, last[a:a + 1024], head_w, li[a:a + 1024], lv[a:a + 1024], use_reentrant=False)
            else:
                l_, d_ = soft_ce(last[a:a + 1024], head_w, li[a:a + 1024], lv[a:a + 1024])
            parts_l.append(l_); parts_d.append(d_)
        lk = torch.cat(parts_l); dk = torch.cat(parts_d)
        losses.append((lk * ok).sum() / ok.sum().clamp(min=1)); drafts.append(dk); valid.append(ok)
        nxt = (t + 2 + k).clamp(max=L - 1)
        tok = dk if own_tokens else ids[nxt]
    return losses, drafts, valid, t, ti


@torch.no_grad()
def evaluate(m, E, scale, files, limit=None):
    m.eval()
    top1 = [[0, 0] for _ in range(K)]
    acc_len, rounds = 0.0, 0
    acc_len_g, rounds_g = 0.0, 0
    n = 0
    for r in iterate(files, val=True, shuffle=False):
        n += 1
        if limit and n > limit:
            break
        Q = r["ids"].numel() - r["r0"]
        for a0 in range(0, Q, 1024):
            # teacher-forced: per-step top-1 agreement with the target's argmax
            _, drafts, valid, t, ti = unroll(m, E, scale, r, train=False, a0=a0, a1=a0 + 1024)
            for k, (d, ok) in enumerate(zip(drafts, valid)):
                want = ti[(t + 1 + k).clamp(max=ti.shape[0] - 1), 0].long()
                top1[k][0] += int(((d == want) & ok).sum()); top1[k][1] += int(ok.sum())
            # own tokens: accepted length of a 7-token draft against the reply's actual tokens
            _, drafts, valid, t, ti = unroll(m, E, scale, r, train=False, own_tokens=True, a0=a0, a1=a0 + 1024)
            alive = torch.ones_like(t, dtype=torch.bool)
            acc = torch.zeros_like(t, dtype=torch.float)
            for k, (d, ok) in enumerate(zip(drafts, valid)):
                want = r["ids"].to(dev).long()[(t + 2 + k).clamp(max=r["ids"].numel() - 1)]
                alive = alive & ok & (d == want)
                acc += alive.float()
            full_ok = valid[-1] if len(valid) == K else torch.zeros_like(t, dtype=torch.bool)
            acc_len += float(acc[full_ok].sum()); rounds += int(full_ok.sum())
            if r["temp"] == 0.0:
                acc_len_g += float(acc[full_ok].sum()); rounds_g += int(full_ok.sum())
    m.train()
    res = {"sequences": n - (1 if limit and n > limit else 0),
           "top1_by_step": [round(a / max(1, b), 4) for a, b in top1],
           "accepted_per_round": round(acc_len / max(1, rounds), 3),
           "accepted_per_round_greedy": round(acc_len_g / max(1, rounds_g), 3),
           "rounds": rounds}
    return res


def save(m, out):
    from safetensors import safe_open
    os.makedirs(out, exist_ok=True)
    with safe_open(BASE + "/model.safetensors", "pt") as f:
        names = list(f.keys())
    msd = m.state_dict()
    missing = [n for n in names if n not in msd]
    assert not missing, missing
    sd = {n: msd[n].detach().to(torch.bfloat16).contiguous() for n in names}
    save_file(sd, out + "/model.safetensors", metadata={"format": "pt"})
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(os.path.join(BASE, f), out)


def main():
    cmd, states = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    E, scale = target_embedding()
    m = load_assistant()
    files = records(states, False)
    log("base drafter:", json.dumps(evaluate(m, E, scale, files)))
    if cmd == "eval":
        return
    out = sys.argv[3]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-5
    m.train()
    head = [p for n, p in m.named_parameters() if n in ("lm_head.weight", "model.embed_tokens.weight")]
    rest = [p for n, p in m.named_parameters() if n not in ("lm_head.weight", "model.embed_tokens.weight")]
    opt = torch.optim.AdamW([{"params": rest, "lr": lr}, {"params": head, "lr": lr * 0.5}], betas=(0.9, 0.95), weight_decay=0.0)
    tok = [x for x in torch.load("/opt/models/n9/tok.pt") if not is_val(x["id"])]
    if os.environ.get("TRAIN_TEST"):
        have = {r["id"] for f in files for r in torch.load(f)}
        tok = [x for x in tok if x["id"] in have]
    n_train = len(tok)
    accum_tokens, steps_total = int(os.environ.get("ACCUM", 16384)), None
    CH = int(os.environ.get("TRAIN_CHUNK", 512))
    tokens_per_epoch = sum(len(x["ids"]) - x["r0"] for x in tok)
    steps_total = max(1, epochs * tokens_per_epoch // accum_tokens)
    warm = min(100, steps_total // 10)
    log(f"train: {n_train} sequences, {tokens_per_epoch} anchors per epoch, {epochs} epochs, ~{steps_total} optimizer steps, lr {lr}")
    step, acc_t, t0, run_loss = 0, 0, time.time(), []
    best = None
    for ep in range(epochs):
        for r in iterate(files, val=False, shuffle=True, seed=ep):
            Qn = r["ids"].numel() - r["r0"]
            for a0 in range(0, Qn, CH):
                losses, _, _, _, _ = unroll(m, E, scale, r, train=True, a0=a0, a1=a0 + CH)
                q = min(Qn, a0 + CH) - a0
                if not losses:
                    continue
                loss = sum(w * l for w, l in zip(STEP_W, losses)) / sum(STEP_W[:len(losses)])
                (loss * q / accum_tokens).backward()
                run_loss.append(float(loss.detach())); acc_t += q
            if acc_t >= accum_tokens:
                cur = lr * min(1.0, (step + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps_total)))
                for g, s in zip(opt.param_groups, (1.0, 0.5)):
                    g["lr"] = cur * s
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1; acc_t = 0
                if step % 50 == 0 or step in (1, 5, 20) or os.environ.get("TRAIN_TEST"):
                    mem = f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if dev == "cuda" else ""
                    log(f"  epoch {ep + 1} step {step}/{steps_total} loss {sum(run_loss) / len(run_loss):.4f} lr {cur:.2e} ({time.time() - t0:.0f}s{mem})")
                    run_loss = []
        res = evaluate(m, E, scale, files)
        log(f"epoch {ep + 1}:", json.dumps(res))
        save(m, f"{out}-ep{ep + 1}")
    log("TRAIN_DONE")


if __name__ == "__main__":
    main()
