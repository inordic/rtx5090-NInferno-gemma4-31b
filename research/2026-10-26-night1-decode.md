# Night 1 (2026-10-25/26): decode tuning

## Result

**Decode is 8–18% faster at the same quality.** The candidate is a new NInferno build (`1a3d564`) plus 7 MTP drafts instead of 8. It went into production on 2026-10-26 at about 06:10 CET with `bench/apply_candidate.sh`; `bench/revert_candidate.sh` rolls it back from the unit-file backups.

On the live unit after the switch: chat decode 118.4 / 114.0 tok/s (greedy / sampled) and prefill 4,154 tok/s at 8K ([`results/results.jsonl`](../results/results.jsonl), label `LIVE fast unit after apply_candidate`).

| Fast profile (QAT-NVFP4, 155K) | Before (night 0) | Candidate |
|---|---|---|
| Decode, chat (greedy / sampled) | 108.6 / 105.6 tok/s | **118.0 / 113.0** |
| Decode after 8K context | 98.1 / 103.6 | **115.3 / 110.7** |
| Decode after 2K context | 117.2 / 128.7 | 119.0 / 126.7 |
| Prefill 2K / 8K / 32K / 64K (tok/s) | 3,047 / 4,248 / 3,449 / 2,591 | 3,101 / 4,137 / 3,438 / 2,579 (same) |
| Agent turn after 17.6K cached tokens | 0.5–0.6 s | 0.5–0.6 s |
| Needles at 32K / 72K / 140K | 9/9 | 9/9 |
| GSM8K / MMLU-Pro / HumanEval | 198 / 150–152 / 161 | 198 / 147 / 161 |

The quality is the same. MMLU-Pro moves between 147 and 152 from run to run on every NVFP4 build; the build with BF16 activations everywhere scored 148. The answer letter also comes from the prefill pass, which the candidate doesn't change.

The same night installed the NInfer web UI (release 0.0.9) on the container: see [`config/ninfer-ui/`](../config/ninfer-ui/).

## What made the candidate faster

All gains are in the decode step, which is where the agent requests seen in production spend their time: a 9.6K-token prompt and about 2,800 tokens of output take about 2 s of prefill and 25 s of decode.

1. **7 MTP drafts instead of 8.** A verify of 8 drafts runs 9 tokens, which falls into the 16-token kernel bucket. 7 drafts runs exactly 8 tokens, which uses the 8-token bucket, and the down and o projections stream 15–17% faster there. In the same session, at 8 / 7 / 12 / 15 drafts:

   | Drafts | Chat (greedy / sampled) | After 8K context (greedy / sampled) |
   |---|---|---|
   | 8 | 108.1 / 105.4 | 89.8 / 95.3 |
   | **7** | **114.3 / 110.0** | **96.3 / 111.0** |
   | 12 | 101.7 / 99.6 | 90.6 / 104.9 |
   | 15 | 96.2 / 94.2 | 79.0 / 88.9 |

2. **QKV projections at 8 tokens** (commit `3036f83`). The two QKV shapes ran their 8-token tile with two row warps by two K warps. Four K warps, the layout gate/up already used, keep more bytes in flight. In the linear benchmark this cuts 16384×5376 from 81.9 to 71.0 µs and 18432×5376 from 90.1 to 73.7 µs, saving about 0.7 ms per step.

3. **Output head at 6–8 tokens** (commit `1a3d564`). The tied Q6 head (262144×5376) ran a 16-column MMA tile at 442 GB/s. The existing sliced 8-column tile reads it at 627 GB/s: 2.50 → 1.76 ms per verify step.

The decode GEMMs at 8 tokens now run at 74–91% of the 896 GB/s bus: gate/up about 818 GB/s, QKV 660–710, down about 730, o 670–740.

## Tried and not adopted

| Experiment | Result | Why not |
|---|---|---|
| W4A4 verify (4-bit activations in the decode pass) | Chat 108.3 / 105.5 vs 108.8 / 108.0 | No speed gain; MMLU-Pro 147 |
| Shortlist drafter head (64K / 96K most frequent tokens) | Chat 109.4 / 108.8 and 113.7 / 108.6 vs 114.2 / 112.3 with the full head | Lost acceptance cancels the cheaper head |
| 16-token schedule variants for down and o | Down −8% at 9 tokens, others worse | 7 drafts avoids that bucket anyway |
| SIMT kernel for the head at 7–8 tokens | 3.0–3.3 ms (worse) | The sliced tile wins |
| Q4 QAT prefill on NInferno's W4A8 tile instead of ggml MMQ | +10% at 8K (1,644 → 1,806), +8% at 32K | Perplexity +2% (90.9 → 92.8) on the quality profile |
| "MSE-scaled" NVFP4 checkpoint (melcheikh) | Identical scores | Its language-model tensors are byte-identical to the absmax repo |

## Agent turns (Hermes): where the 0.3 s goes

A cached agent turn with a few dozen new tokens takes about 0.3 s to its first token: 296 ms for 38 new tokens, 348 ms for 230, 571 ms for 865.

An nsys trace shows the GPU busy for the whole turn, so this is GPU work, not CPU-side tokenizing. A turn does three things:
- prefills the new tokens against the cached context;
- copies a state snapshot to host memory (about 45 ms) and restores one (about 16 ms), because at 155K context there is no room for a device-side cached state;
- runs the first decode step.

With a shorter `--max-context`, the plan keeps two cached states on the device, and turns get 80–100 ms faster. At 100K, a 38-token turn drops from 0.30 to 0.20 s and a 230-token turn from 0.35 to 0.29 s. If Hermes rarely goes past 100K tokens, that trade is worth it:

```bash
ssh root@<host> 'sed -i "s/--max-context 155000/--max-context 100000/" /etc/systemd/system/gemma4-31b-fast.service && systemctl daemon-reload && /root/gbench/deploy_fast.sh'
```

The real fix is engine work: overlap the snapshot copies with compute.

## Tool calling works

OpenAI-style tools work on the deployed unit; I checked them because Hermes and the web UI's tool loop need them.
- **Plain request:** a correct `get_weather({"city":"Oslo"})` call with `finish_reason: tool_calls`. After the tool result came back, it gave a correct answer. Each step took 0.28 s.
- **Streaming:** two parallel calls (Bergen and Oslo) arrived as streamed `tool_calls` deltas.

## The quality profile (262K, QAT Q4)

The candidate binary works for it too: 7 drafts give chat 106.8 / 99.7 against 104.5 / 99.4 with 8. On the candidate it also passes the long-context checks:

- **Needles:** 2/2 at 185K and 231K tokens.
- **Prefill at those lengths:** 613 and 528 tok/s, against 380–468 before the FP8 attention work.
- **Agent turns:** 0.63–0.66 s after the first.

Its short-prompt prefill stays near 1.5K tok/s, because ggml's MMQ runs at about 0.33 of INT8 peak on this card.

Research ([`2026-10-25-qat-q4-fast-prefill-design.md`](2026-10-25-qat-q4-fast-prefill-design.md)) found the way past that: a Q4_0-specialised INT8 kernel cloned from NInferno's I8 route. The QAT Q4_0 nibbles feed the MMA directly, with no decode step. The estimate is 2.0–2.6K tok/s at Q4 quality, which would give a profile with Q4's MMLU-Pro 164 and about 2.5K prefill. It is a multi-hour kernel job, not started yet.

## Code

NInferno `kvshare-main` on CT116, overnight commits:
- `3036f83`: QKV 8-token schedule.
- `1a3d564`: Q6 head 8-column tile.

The working tree is clean. The env-gated experiments are saved as patches in `/root/gbench/patches/experiments`. All patches are also in `patches/` in this project (0006, 0007, `experiments/`).
