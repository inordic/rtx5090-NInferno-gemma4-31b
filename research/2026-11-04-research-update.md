# Research update (2026-11-04): what is new, and the plan for night 5

## Result

Eight research passes looked at what changed since 25 Oct: the engine's upstream, Gemma 4 checkpoints, other models, drivers and CUDA, speculative decoding, long context, prefill kernels and quality methods. Nothing changed on the running setup.

- **No outside release moves our numbers.** No new Google weights or drafter for the 31B, no engine that beats NInferno on this card, no driver with a speed claim for GeForce Blackwell, and no upstream change to a kernel on our path since 31 Oct.
- **The best finds are in our own code.** Layer 59 computes attention and MLP for every prompt token and then throws them away, and the GEMM tile was tuned for 170 SMs, where this card has 82.
- **Upstream's value now is stability.** Six fixes from 2 Nov turn the kind of engine exit we have had into a failed request.

Gains below are estimates unless they say measured. The night tests them; none is proven on this box yet.

| Candidate | Expected | Evidence | Night 5 |
|---|---|---|---|
| **Patch 19: stop the last layer early on mid-prompt calls** | prefill +2.6% at 8K, +4.6% at 32K, +8% at 130K, identical output | read from our code by two passes separately; code paths checked by hand | W2, W3 |
| **Six upstream fixes, `kvshare-up4`** | no speed; exits become failed requests, and a tool result ending in an image stops failing | upstream's tests on other cards | W2, W3 |
| **Prefill chunk 1280 or 1792** | prefill +6.4% at 8K, +4.9% at 32K, about 10K less context | model of the 82-SM wave tail: N=5376 at chunk 1024 is 168 tiles, two waves and 4 tiles left over; day-1 chunk runs fit it | W1 |
| **Skip-softmax in the 512-wide FP8 kernel** | perhaps +15–25% at 130K, with a quality risk | NVIDIA's blog and MLSys paper; no sm_120 or Gemma 4 number exists | W4, counting only |
| **`CudaNoStablePerfLimit` profile (P0 instead of P2)** | decode up to +3%; prefill must not drop | measured on an RTX 4080 by a third party: decode +3.6%, prefill flat; nothing on Blackwell | W5 |
| **`ninferno-perplexity --kld`** | a test that sees quality changes far smaller than our 280 questions can | already in our base, never built here | W6 |
| **Assistant in NVFP4** | decode +2.7%, about 4.8K more context | upstream measured +2.7% on a desktop RTX 5090 | W7 |
| **CUDA 13.4 rebuild, same driver** | unknown; nobody has published sm_120 timings across toolkits | upstream validates 13.4 | W8, if disk allows |

## 1. The engine upstream

Master has 2,274 commits on top of our base `8359583` (212 on its first-parent line). From our last review at `a8beb1c` to `1d3b43e` there are 88 merges, and none changes a kernel, route or table on dense Gemma 4 with NVFP4 and MTP on sm_120.

The six picks, on `kvshare-up3`, with patch 12 reverted because #318 carries its fix:

| PR | What it does for us |
|---|---|
| #318 | a failure contained to one request answers 500, and the engine resets its sequences and cache instead of exiting; three in a row with no token between them still restart the server |
| #327 | pages an owner shared with the active sequence return to the request when the owner is released; probably the root cause of the exit patch 17 guards against |
| #324 | a KV trim to the committed frontier keeps a shared full tail page |
| #325 | a Chat tool result ending in an image no longer answers 500 on Gemma 4; cut-off calls stay out of the text |
| #326 (`c6e54d71` only) | the capture assessment counts a dropped checkpoint as freed |
| #319 | a refusal keeps the connection |

They applied with one conflict, in a list of includes. Patches 09 and 15 turn out to be the same code as upstream #321. Night 5 counts `publishes privately` lines on this build; if it stays at zero, patch 17 may be able to go.

A full rebase looks possible (8 conflict hunks, only patch 04 is real work), but it brings no speed, so it can wait for a quiet week. Upstream removed adaptive draft depth, so that item leaves our list.

## 2. Prefill

No engine beats NInferno on this card. No one publishes a long-context NVFP4 prefill number for Gemma 4 31B on 32 GB or less.

Our W4A4 GEMMs run at about 450 TFLOPS at chunk 1024 and 490–530 at chunk 2048, the same class as the best published sm_120 kernels. The 4.4K tok/s figure looks lower because GEMMs take only 17–58% of prefill time. The largest gap is the 512-wide attention: about 90 TFLOPS, roughly 28% of the FP8 rate, and about 72% of prefill at 130K. It is still the fastest D=512 prefill we could find on sm_120: no public kernel handles FP8 at that width. Those shares assume a clock of 1.88 GHz, which W1's trace will check.

**Patch 19.** The last layer is a full-attention layer no later layer attends through. On a mid-prompt call only its keys matter, and those are written before its attention starts. The patch stops there, but only on the staged prompt path. Two other paths read what it skips, so they keep the full pass:

- a rebuild reads every call's tail hidden;
- a causal score reads every column.

A call ending on a capture's frontier also keeps it, because its tail goes into the checkpoint. It is opt-in (`NINFERNO_GEMMA4_SKIP_LAST_LAYER=1`).

FP4 attention is not worth porting: quality falls without retraining, and no kernel handles D=512.

## 3. Long context

- **262K with vision** is about 1.07 GB short, not 2.2 GB. The gap is the vision tower. Moving it off the device is a watch item; the agent's largest prompt so far is 138K.
- **4-bit KV:** no new result at 128K or above, and upstream has none for Gemma 4. A paper of 2 Nov (FlagRot) reports that whole-head rotation raised NVFP4 KL by 82% on Gemma 3 27B, so our 25 Oct design of a rotated cache would have to change.
- **Beyond 128K** we found no independent quality measurement of Gemma 4.
- Before any approximate attention ships, our needle test needs to get stronger; one 140K needle already fails on every NVFP4 build.

## 4. Speculative decoding

Live acceptance of 43% is normal: upstream measures 42.7% on its own corpus. The log shows no step-downs and no watchdog trips in 690 requests; 344 of them used n-gram copy rounds.

- **DFlash:** skipped. The server runs Gemma with MTP only, and a DFlash drafter would cost roughly 86K of context.
- **A drafter trained on our traffic:** not realistic overnight; the tooling is missing and the published live gain is +3–7%.
- **Decode's share:** about 40% of GPU-busy time on the live traffic, so a 10% faster decode saves about 65 ms on a median turn.

## 5. Quality

- **The gap to BF16 is unknown.** Published BF16 MMLU-Pro (84–87) is with reasoning; ours is a single direct letter. Direct-letter results are a poor proxy for agent quality, and runs of the same weights changed 25 of 280 answers.
- **The NVFP4 block scales were never searched.** The "MSE" NVFP4 repo we tried on 26 Oct is byte-identical to plain absmax scaling. A true search in `hlquant.py` is CPU work, keeps the plain format and cuts weight error about 10%, with the task gain unknown. It needs Google's BF16 weights on disk.
- **W4A8, Hadamard rotation and MR-GPTQ:** skipped; published results show no gain, or a loss on this model.
- **Gemma template PR #147** (an empty thought after a tool response) targets empty replies. We had 0 in 690 requests, so it stays out of the night.
- **Tool markup returned as text**, at about 0.5%, matches what vLLM reports for this model.

## 6. Models

No Gemma 4 31B checkpoint shows a measured lead over LilaRest's NVFP4-turbo at our size; the better builds are 22.5 GiB or more. Fine-tunes publish little evidence. The best is TermGrade (Terminal-Bench +2.1 over five runs), but it would change only about 0.012% of NVFP4 codes.

Among other models, Qwen3.8-27B is the only candidate. Third-party tests show it ahead on agent and coding tasks, with thinking off against Gemma's thinking on. On this card it would fit about 87K of context at 8-bit KV, far from our 202K, and no one has published its NInferno prefill. It stays on the list until it can pass a speed test.

## 7. Drivers and power

The enforced power limit is **150 W** (default 95 W, maximum 175 W), not the 175 W of our earlier notes. That suggests Dynamic Boost is off, but we have never traced power or clocks under load. W1 does that first: if prefill sits on the power cap, more power is worth a look on the host. If it doesn't, that idea can go.

No driver after 595.58.03 claims faster GeForce Blackwell compute. Neither NInferno's sm_120 path nor our shapes use cuBLASLt or CUTLASS, so their updates change nothing for us.

## 8. Night 5

Runner and tools: [`bench/runs/night5/`](../bench/runs/night5/). Windows run behind the idle gate, the fast unit comes back after each, a window stops when its server is gone, and none starts after 05:30 CET.

| Window | What | Passes if |
|---|---|---|
| W1 | production build: speed and a power trace; chunk 1280 and 1792 at the largest context that fits | a chunk gains at every length, for a context cost worth paying |
| W2 | `apps-up4`: greedy answers with patch 19 off, off again, on (cold, repeated and follow-up turns); speed | off and on give the same tokens; prefill rises |
| W3 | `apps-up4` with patch 19 on: unit tests, tools, vision, image tool result, turns, heads, agent loop, three replays, 300 stress steps, needles, MMLU-Pro, GSM8K, hard50 | no exit, same quality and speed elsewhere |
| W4 | counting build: share of 512-wide score tiles under 2^-6 … 2^-14 at 8K–131K | at least 25% at 2^-10, or the idea stops |
| W5 | P0 profile, removed afterwards | P0 shows, decode rises, no prefill figure falls |
| W6 | KLD: production, KV alone and A4 alone against a weights-only reference | shows where the error sits |
| W7 | assistant in NVFP4 | decode up, acceptance the same, context up |
| W8 | CUDA 13.4 build of the same tree, only with 8 GB of disk free | speed up, same tokens |

Production changes only after the results, and only with an explicit OK.
