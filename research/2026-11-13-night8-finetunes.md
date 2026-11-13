# Night 8 (2026-11-13): two creative fine-tunes for a companion agent

## Result

Production did not change. Two popular creative fine-tunes of Gemma 4 31B went through the full battery in production's format: [zerofata/G4-MeroMero-v2-31B](https://huggingface.co/zerofata/G4-MeroMero-v2-31B) and [TheDrummer/Artemis-31B-v1.2](https://huggingface.co/TheDrummer/Artemis-31B-v1.2). The question was whether either gives a companion-style agent a better voice without costing quality, speed or tool reliability.

- **No clear win.** Both stay within noise of production on quality, and neither is better.
- **MeroMero v2** matches production's speed, but in one tool test it invented an argument instead of asking for it.
- **Artemis v1.2** passes every tool test and sounds a little livelier. It decodes 5–10% slower, because Google's drafter accepts fewer of its tokens.

| | Production (LilaRest NVFP4-turbo) | MeroMero v2, imw | Artemis v1.2, imw |
|---|---:|---:|---:|
| Prefill 2K / 8K / 32K | 4,192 / 4,732 / 3,710 | 4,197 / 4,678 / 3,672 | 4,171 / 4,672 / 3,640 |
| Chat decode, greedy / sampled | 113.6 / 110.6 | 116.3 / 110.7 | 108.2 / 99.2 |
| Drafter acceptance on hard50 | 61.2% | 62.7% | 58.4% |
| MMLU-Pro (280) / MMLU (285) | 178 / 249 | 175 / 246 | 174 / 244 |
| GSM8K (200) / HumanEval (164) / hard50 (50) | 197 / 161 / 42 | 198 / 161 / 41 | 198 / 160 / 43 |
| Total of 979 items | 827 | 821 | 819 |
| Tool-call leaks; agent loop errors | 0; 0 | 0; 0 | 0; 0 |
| Vision; context | 7/7; 203,136 | 7/7; 203,136 | 7/7; 203,136 |
| Companion-agent set (10 checks) | 10 | 9 | 10 |

Production's MMLU-Pro and MMLU were measured the same morning on the same harness. Its speed rows come from night 5, on the same build and flags.

## How

- **Build:** each BF16 checkpoint (62.6 GB, Google's full layout with the vision tower) was quantized with [`hlquant2.py`](../bench/hlquant2.py) importance-weighted rounding. Each used the public importance matrix of that exact model (mradermacher's for MeroMero, bartowski's for Artemis), about 3 minutes on the GPU. Then it was converted as production's model (BUILD 3b).
- **Serving:** production's binary (`apps-up4`), flags and NVFP4 drafter. Both load at 203,136 tokens with vision and the same memory plan.
- **Battery:** [`battery8.sh`](../bench/runs/night8/battery8.sh) runs verify, tool-call leaks, vision, the agent loop, speed, the four quality suites, hard50, the eight style chats and the companion-agent set.

**The companion-agent set** ([`companionfit.py`](../bench/runs/night8/companionfit.py)) uses a synthetic persona with moods and seven tools, greedy, thinking off. It checks ten things:
- answering in the user's language (Swedish);
- no tool calls in small talk;
- reaching out with a message after three quiet days;
- reading mail, then flagging the urgent bill;
- not following an instruction planted in an email (forward and delete);
- asking for a missing reminder time instead of inventing one;
- a two-step calendar lookup, then a reminder one hour before the first event;
- two weather calls in one turn.

MeroMero failed the missing-time check: it set the reminder for the current time.

**Voice** ([`voice.py`](../bench/runs/night8/voice.py)): six companion prompts, three samples each at temperature 0.9.

| | Production | MeroMero v2 | Artemis v1.2 |
|---|---:|---:|---:|
| Words per reply | 110 | 102 | 114 |
| Emoji per reply | 0.2 | 0.1 | 0.8 |
| Distinct openings across the three samples (of 18) | 12 | 15 | 14 |

At temperature 0 all three sound nearly the same under a strong persona prompt; MeroMero's greedy replies often match production's word for word. Sampled, both fine-tunes vary their openings more. Artemis is a little more colourful and joking. Both fine-tunes occasionally invent things they have not done ("I was looking through your photo library earlier").

Results: [`n8-finetunes.out`](../results/runs/n8-finetunes.out). The replies themselves stay local.

## Next

- **Artemis** is the only candidate. A switch would want a few days beside production on real conversations first, and the 5–10% decode cost accepted.
- **The companion set is small.** One failure out of ten is a signal, not a rate. More tool-argument edge cases would firm it up before any switch.

## Update, 13:02 CET: Artemis deployed

The agent's owner chose Artemis without the shadow run, accepting the decode cost. It replaced the previous model's weights in the same unit, with the same flags and model id. After the restart it passed the same checks as in the battery: 0 tool-call leaks, vision 7/7, agent loop without errors, and companion set 10/10. Its live traffic is the real test now. See the [CHANGELOG](../CHANGELOG.md).
