"""kltext10.py: three KL texts from Artemis's own held-out replies, for comparing activation scales
(sequences whose id hashes to 0-2 of 20; night 10's activation peaks exclude them; night 9's peaks, taken over its
first 208K tokens, may include some of night 9's):
  kl_chat.txt   night 9: UltraChat, oasst2, Swedish, companion, tool calls
  kl_code.txt   night 10: code, maths, English chat
  kl_agent.txt  night 10: agent episodes (long system prompt, tool calls and results)
Each sequence is the exact token sequence that was served, decoded with its special tokens; capped per text."""
import hashlib, sys
import torch
from transformers import AutoTokenizer

CAP = {"chat": 60000, "code": 50000, "agent": 50000}
tok = AutoTokenizer.from_pretrained("/opt/models/n8/A2/bf16")


def held(rid):
    return int(hashlib.md5(rid.split("~")[0].encode()).hexdigest(), 16) % 20 in (0, 1, 2)


sets = {"chat": [x for x in torch.load("/opt/models/n9/tok.pt") if held(x["id"])],
        "code": [x for x in torch.load("/opt/models/n10/tok.pt") if held(x["id"]) and x["cat"] in ("code", "maths", "english")],
        "agent": [x for x in torch.load("/opt/models/n10/tok.pt") if held(x["id"]) and x["cat"] == "agent"]}
for name, xs in sets.items():
    parts, n = [], 0
    for x in xs:
        if n >= CAP[name]:
            break
        ids = x["ids"][1:] if x["ids"][0] == tok.bos_token_id else x["ids"]
        parts.append(tok.decode(ids, skip_special_tokens=False))
        n += len(ids)
    open(f"/root/gbench/n10/kl_{name}.txt", "w").write("".join(parts))
    print(f"kl_{name}.txt: {len(parts)} sequences, {n} tokens")
