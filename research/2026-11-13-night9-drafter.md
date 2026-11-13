# Night 9 (13 Nov 2026): the MTP drafter retrained on Artemis

The fast profile switched to Artemis v1.2 at 13:02, with everything else unchanged, Google's MTP assistant included. Artemis is a fine-tune with its own style, and the assistant accepts fewer of its drafts than it did on LilaRest's build: chat decode 108.2 / 98.1 tok/s against 113.7 / 110.6. Tonight's question: does fine-tuning the assistant on Artemis's own text and hidden states win that back?

**Result:** a small gain that is real on conversation and tool calls, and none on code or maths. It went live at 20:37 on the owner's request ([CHANGELOG](../CHANGELOG.md)).

| Measured on production's binary and flags, one run each | Google's assistant | Retrained |
|---|---:|---:|
| Held-out prompts, all (78): decode tok/s / drafts accepted | 83.9 / 27.0% | **86.1 / 28.1%** |
| greedy (39) | 90.1 / 30.0% | **94.2 / 31.9%** |
| sampled (39) | 78.9 / 24.6% | 79.6 / 24.9% |
| ultrachat (English) | 80.1 / 25.2% | 83.6 / 26.8% |
| tool calls | 148.0 / 58.3% | 152.1 / 61.1% |
| Swedish | 76.1 / 23.2% | 79.3 / 24.7% |
| oasst2 (several languages) | 86.5 / 28.1% | 85.1 / 27.4% |
| Chat benchmark (code and technical, 10 × 512), greedy / sampled tok/s | 108.2 / 98.1 | 108.9 / 98.2 |
| hard50 (maths and MMLU-Pro), drafts accepted | 58.4% | 57.9% |

The two assistants ran one after the other, not alternating. To separate the effect from run-to-run noise, the held-out requests were paired. Of the 32 greedy requests with at least 10 draft rounds, the retrained assistant accepted more drafts on 24 and fewer on 8 (sign test, p ≈ 0.007). 29 of those 32 produced the same output length both times, so the text was nearly identical. On the sampled requests it was 18 against 12, which is not significant.

The chat benchmark's greedy outputs were token-for-token the same length (4,856 tokens) with both assistants, and the difference there is +0.6%. The training data had almost no code, and the gain doesn't carry over to code or maths.

## Data

- **Prompts:** [`bench/runs/night9/gen.py`](../bench/runs/night9/gen.py).
  - Public sets: UltraChat, OpenAssistant oasst2 in twelve languages, and single-turn Hermes function-calling prompts.
  - Prompts Artemis wrote itself: Swedish questions, and casual companion messages in English and Swedish, for a synthetic persona with seven tools.
  - No real conversations.
  - About 60% of the material is English, 24% Swedish and 16% the other oasst2 languages.
- **Replies:** each prompt was answered by the production server through its normal API. Requests went out only while the server was idle. The temperature was 0, 0, 0.7 or 1.0 in turn, with up to 1,024 tokens (512 for companion messages).
  - 1,930 replies in 4 h 20 min, 1.84M tokens in all, 1.2M of them Artemis's own.
  - A third hit the token limit; a cut-off reply is still valid training text.
- **Tokenizing:** [`fwd.py tokenize`](../bench/runs/night9/fwd.py) renders the request with production's chat template and appends the reply as served. All 1,930 matched the server's prompt token count.

## The target's states

The assistant reads the target's last hidden state and the K/V of its last sliding and last full attention layers (58 and 59). [`fwd.py run`](../bench/runs/night9/fwd.py) runs Artemis one decoder layer at a time, with the weights streamed from disk and dequantized on the GPU from the same importance-weighted NVFP4 weights production serves, and BF16 activations. It keeps, for every position:
- the final hidden state;
- the top 32 log-probabilities;
- the layer-58/59 K/V.

That took 37.5 minutes for 1.84M tokens in six passes, and wrote 61 GB.

**Check:** the pass's greedy token agrees with the token the server actually produced at 97.9% of 590,774 greedy positions. The rest is activation quantization and kernel numerics.

## Training

[`bench/runs/night9/train.py`](../bench/runs/night9/train.py):
- **Model:** fine-tunes `google/gemma-4-31B-it-assistant` (four layers, hidden 1024) in FP32 under BF16 autocast. Inputs and masks are exactly NInferno's MTP round.
- **Steps:** from each reply position, seven chained draft steps. The first feeds the target's embedding of the next token and its last hidden state; later steps feed the true next token and the assistant's own carried state.
- **Loss:** each step learns the target's top-32 distribution at its position (soft cross-entropy), weighted 0.85^k.
- **Optimizer:** AdamW (0.9, 0.95), learning rate 5e-5 with a cosine decay (half for the tied head), 4,096 positions per step.
- **Data:** 1,825 sequences (1.12M positions) for training. 105 sequences are held out by a hash of the prompt id, and none of them was trained on.

The first run ran out of GPU memory on its first step: autograd kept all seven steps × four layers of attention internals. It now recomputes each step in the backward pass and takes 512 positions at a time, and peaks at 14.7 GiB. Two epochs took 52 minutes.

Offline, on the held-out sequences, with the assistant drafting from its own tokens:

| | Base | Epoch 1 | Epoch 2 |
|---|---:|---:|---:|
| Tokens accepted per round | 2.152 | 2.237 | **2.281** (+6.0%) |
| same, greedy replies | 2.342 | 2.423 | 2.465 (+5.3%) |
| Step-1 top-1 agreement | 71.9% | 73.9% | 74.3% |
| Step-7 top-1 agreement | 54.7% | 56.6% | 58.0% |

Epoch 2 was converted with production's recipe (`gemma4_assistant_published`, NVFP4), to a file the same size as the current assistant: 296,346,628 bytes.

## Two side results

- **Long-prompt prefill on a rested GPU:** 1,720 tok/s on the 128,932-token needle prompt, which passes. Right after this afternoon's deployment, the same prompt measured 1,562 tok/s. The gap was heat, not the model.
- **Artemis's activations against the calibrated scales:** the forward pass recorded the peak input of every projection over its first 208,577 tokens. Production's NVFP4 uses LilaRest's activation scales. Artemis exceeds them in 100 of 240 places, nearly all in the MLP, up to 12× in the down projections of layers 9–15, 27–29 and 56, so those activations clip. The 97.9% agreement above already includes that clipping. The attention output projection's peaks sit at about 1% of its calibrated value, which points to a different calibration convention, so this comparison is a first look only. A calibration for Artemis itself is a separate experiment ([review, point 8](2026-11-13-review.md#8-quantization-calibrated-for-artemis)). The peaks are in [`results/runs/n9/amax-imw.json`](../results/runs/n9/amax-imw.json).

## Next

- **Training data:** add code and maths prompts, the areas where the gain didn't carry over, and more multi-turn agent text.
- **Live trial:** the retrained assistant can be tried on production with a one-line unit change and taken back as quickly. A draft model can't change the target's output beyond numerical noise.
- **Measurement:** alternating A/B runs with the spread reported, and a draft-depth sweep (K = 3/5/7) for both assistants on Artemis ([review, point 6](2026-11-13-review.md#6-draft-depth-for-artemis)).

Files: [`bench/runs/night9/`](../bench/runs/night9/) (the scripts) and [`results/runs/n9/`](../results/runs/n9/) (the runner outputs, both measurement files, the per-request held-out rows, the training logs, the activation peaks); rows `N9 old` and `N9 new` in [`results/results.jsonl`](../results/results.jsonl), `N9 rested needle` in [`results/needle.jsonl`](../results/needle.jsonl).
