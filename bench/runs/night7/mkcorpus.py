"""chat.txt: harmless prompts with the production model's own answers, in Gemma 4's chat format (thinking off),
for KL divergence on in-distribution text. Kept local. Prints the token count with and without special-token parsing."""
import json, sys
from tokenizers import Tokenizer
D = "/root/gbench/n7"
convs = []
for s in ("xstest_safe", "orhard"):
    for l in open(f"{D}/refusal/A0.{s}.jsonl"):
        r = json.loads(l)
        if r["reply"].startswith("<<error"):
            continue
        convs.append(f"<|turn>user\n{r['prompt']}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{r['reply']}<turn|>\n")
text = "".join(convs)
open(f"{D}/chat.txt", "w").write(text)
tok = Tokenizer.from_file("/opt/models/hl/bf16/tokenizer.json")
n_special = len(tok.encode(text, add_special_tokens=False).ids)
print(f"chat.txt: {len(convs)} conversations, {len(text)} chars, {n_special} tokens with special tokens parsed")
