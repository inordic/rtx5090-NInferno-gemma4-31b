# Changelog

Every change to the running setup, newest first. Each entry names what changed on the box, why, the measurement behind it, and how to roll it back. The README, BUILD.md, configs and results are updated in the same commit. Times are CET.

## 2026-10-28 06:38: fast unit on the rebased build, with the prompt-cache flags

- **Changed:** `gemma4-31b-fast` ([unit](config/systemd/gemma4-31b-fast.service)) now runs the night-3 proposal ([report §4](research/2026-10-28-night3.md#4-proposed-production-change)):
  - **Binary:** `/opt/NInferno-up/build/apps-up1`, branch `kvshare-up` at `afd15fa`: our patches 01–16 on upstream NInferno `8359583`. It includes patch 15, the fix for the engine exit of 27 Oct.
  - **New flags:**
    - `--auto-tools-shared-prefix --auto-shared-prefix-requested` (patch 16): a new session's system head and tool list compete for the cache like a marked prefix, instead of only using spare room;
    - `--host-state-slots 18 --host-kv-mib 8192`: NInferno's default host cache, without the cap at 30% of RAM (11 states before).
  - **Restart:** 06:38, with no request in flight; 8080 answered again after 30 s. The plan is the same as before: 202,112 tokens × 1 lane.
  - **Unchanged:** the model, drafter, context, vision, KV settings, environment switches and chat template. The max-context unit stays on `apps-final7`.
- **Why:** in a replayed agent day with synthetic prompts, new sessions waited 1.5 s instead of 4.4 s on average, and the whole replay took 109 s instead of 192 s. Speed, quality and the regression checks matched production ([night-3 report](research/2026-10-28-night3.md)).
- **Measured on the live unit after the switch:**
  - [`verify.py`](bench/verify.py) passed: chat, streaming, and agent turns of 0.52–0.53 s with 17.6K cached;
  - a streamed tool call returned `tool_calls`;
  - [`textcalltest.py`](bench/textcalltest.py): 0 leaks;
  - [`visiontest.py`](bench/visiontest.py): 7/7;
  - [`agenttest.py`](bench/agenttest.py): 30 steps, 28 calls, no markup as text, no error;
  - [`headprobe.py`](bench/headprobe.py): new sessions sharing a 24.4K-token system head and 20 tools reused 24,192 tokens each and answered in 0.44–0.50 s. A new session of that size took 7–8 s before;
  - memory: server RSS 24.3 GiB (about 16 GiB before), 23 GiB still available in the container; GPU 23.9 GB, as before;
  - the web UI's `/health` reports `max_context 202112, images true`.
- **To watch:** how often the agent's own new sessions now reuse the cache. Its system heads may also change from run to run, and no cache setting fixes that.
- **Rollback:** restore `/etc/systemd/system/gemma4-31b-fast.service.bak-20261028-053845` (`apps-final9`, no cache flags), then `systemctl daemon-reload` and [`bench/deploy_fast.sh`](bench/deploy_fast.sh).

## 2026-10-28 01:49: night 3 tests (running setup unchanged)

- **Ran:** four sessions from 23:52 to 01:49, behind the idle gate, the fast unit back after each. They covered the engine exit of 27 Oct, new agent sessions and the prompt cache, and our patches rebased on upstream NInferno. Results and a proposed change are in the [night-3 report](research/2026-10-28-night3.md).
- **On the box, not serving:**
  - NInferno commits `239b6cf` (patch 15, the fix for the 27 Oct exit) and `8ab8935` (patch 16, two opt-in cache flags) on `kvshare-main`, built as `apps-final10`;
  - all sixteen patches rebased on upstream `8359583` as branch `kvshare-up` in `/opt/NInferno-up`, built as `apps-up1`.
- **Proposed, waiting for an OK:** the fast unit on `apps-up1` with `--auto-tools-shared-prefix --auto-shared-prefix-requested --host-state-slots 18 --host-kv-mib 8192` ([report §4](research/2026-10-28-night3.md#4-proposed-production-change)).
- **Unchanged:** 8080 runs `apps-final9` as before.

## 2026-10-26 22:43: fast profile on LilaRest NVFP4-turbo with vision, 202K context

- **Changed:** `gemma4-31b-fast` ([unit](config/systemd/gemma4-31b-fast.service)) now serves the build recommended by night 2:
  - **Model:** LilaRest's NVFP4 of Google's BF16 Gemma 4 31B, with its Q/K/V packed (patch 13) and the vision tower of Google's original model (`/opt/models/dl2/turbo-qkv-vision-nvfp4.r31.ninferno`, [BUILD 3b](BUILD.md#3b-the-fast-profiles-target-lilarest-nvfp4-turbo-qkv-packed-with-vision)).
  - **Drafter:** Google's regular assistant in Q8 (`assistant.q8full.ninferno`), 7 drafts.
  - **Binary:** `build/apps-final9`, NInferno `12bafc3`: patches 10–14 on top of production's 01–09 ([`patches/ninferno/`](patches/ninferno/)). Patch 14 is new: `/v1/models` reports `capabilities.images`, which the web UI needs before it sends an image.
  - **Flags:** `--vision --max-context 202112 --sliding-kv-dtype fp8`, `NINFERNO_GEMMA4_EMBED_HOST=1`, `NINFERNO_GEMMA4_WIDE_PREFILL_MIN_T=64`. Chunk, FP8 KV, K-only KV, the GeGLU split and the agent template are as before.
  - **Restarts:** 8080 restarted at 22:40 (`apps-final8`) and 22:43 (`apps-final9`), each at a moment with no request in flight.
  - **Unchanged:** the max-context unit stays on `apps-final7`.
- **Why:** it is the one build that meets all four goals. Prefill doesn't drop, context grows, quality rises, and vision works ([night 2 §7](research/2026-10-26-night2.md#7-follow-up-the-recommended-build-end-to-end-22102229-cet)).
- **Measured**, against production in the same session:

  | | Before | After |
  |---|---|---|
  | Prefill 2K / 8K / 32K / 64K (tok/s) | 3,126 / 4,345 / 3,469 / 2,571 | 3,676 / 4,361 / 3,450 / 2,562 |
  | Context | 155,000 | 202,112 with vision |
  | MMLU-Pro (of 280) | 147–152 | 177–178 |
  | Images | none | 7/7 |
  | Chat decode | 118.0 / 115.8 | 109.6 / 110.9 tok/s, 5–7% lower |
  | Agent turns | 0.28–0.34 s | 0.24–0.28 s |

  After the switch, on the live unit:
  - [`verify.py`](bench/verify.py) passed: chat, streaming, and agent turns of 0.53 s with 17.6K cached;
  - a streamed tool call returned `tool_calls`;
  - [`visiontest.py`](bench/visiontest.py) scored 7/7;
  - [`agenttest.py`](bench/agenttest.py): 31 steps, 29 calls, no markup as text, no error;
  - [`textcalltest.py`](bench/textcalltest.py): 0 leaks;
  - the web UI's `/health` reports `max_context 202112, images true`.
- **Rollback:** restore `/etc/systemd/system/gemma4-31b-fast.service.bak-20261026-214015` (`apps-final7`, QAT-NVFP4, QAT assistant, 155K), then `systemctl daemon-reload` and [`bench/deploy_fast.sh`](bench/deploy_fast.sh).

## 2026-10-26 21:45: night 2 tests (running setup unchanged)

- **Ran:** ten batches from 18:43 to 21:45, the fast unit back after each. They covered the context levers, speed levers, n-gram settings, vision, other NVFP4 builds and uncensored builds. Results and candidates are in the [night-2 report](research/2026-10-26-night2.md).
- **On the box, not serving:**
  - NInferno commits `633a7e0`, `352aeba`, `28fdaac` and `91d0b18` ([candidate patches 10–13](patches/ninferno/));
  - test builds `apps-final5`, `apps-final6`, `apps-visfix`;
  - the packed LilaRest turbo artifact.
- **Follow-up, 22:10–22:29:** the recommended build end to end, LilaRest turbo with Q/K/V packed and Google's vision tower, on `apps-final8` ([report §7](research/2026-10-26-night2.md#7-follow-up-the-recommended-build-end-to-end-22102229-cet)).
- **Unchanged:** 8080 runs `apps-final7` with the agent template, as before.

## 2026-10-26 14:24: the fast unit starts at boot

- **Changed:** `systemctl enable gemma4-31b-fast`, so 8080 comes back with the current setup when the container restarts: `apps-final7`, the agent template, 155K context. The web UI (`ninfer-ui`) was already enabled. Every other model unit on the box stays disabled, so nothing else takes 8080 or the GPU at boot.
- **Boot behaviour:**
  - the unit starts after `network.target`; the GPU device nodes are passed in from the host and exist when the container starts;
  - a start that fails (GPU still in use by another container, say) is retried every 10 s; systemd's start limit (5 in 10 s) cannot trip at that interval.
- **Not tested:** an actual container reboot.
- **Rollback:** `systemctl disable gemma4-31b-fast`.

## 2026-10-26 12:49: agent tool calls: an updated chat template, and a crash on resumed conversations fixed

- **Changed, template:** both units pass `--chat-template /opt/models/templates/gemma4-agent.jinja` ([`config/templates/`](config/templates/)). This is Google's Gemma 4 template of 2026-07-09 with two agent changes:
  - an assistant message's text goes before its tool calls, the order the model writes them;
  - a turn whose tool results end the conversation stays open.

  Before, prompts used the template stored in the QAT-NVFP4 artifact, an older one from its quant's repo.
- **Changed, engine:** both units run `build/apps-final7`: NInferno `e9203c5`, which is `6ff9931` plus [patch 09](patches/ninferno/). Same flags otherwise. The fast unit restarted at 12:44 (diagnostic build) and 12:47 (the fix), each time back in a few seconds; the unit files were switched to `apps-final7`, the same binary, at 12:49 without a restart.
- **Why:** the server log of this morning's agent sessions showed two faults.
  - **Tool markup returned as text, five times (10:09–10:50):** four "malformed structure" and one "trailing content", so the agent got markup instead of a call.
    - The old template closed the model's turn after an assistant message with text and calls, then wrote every later call and result of the conversation outside any turn.
    - Google's July template fixes that, but still ends the prompt on a closed turn when the last assistant message had text.
  - **The engine exited twice (10:37, 10:43)** on agent requests with 82 and 100 messages, with `materialized sequence does not match its active entitlement`. Each crash cost the request and about 40 s of 8080.
    - A request that takes over a cached conversation counted a checkpoint another cached conversation still shared as its own; the ownership count did not.
    - The diagnostic build showed it: expected `host_state 1`, actual 0, reuse `PrivateEndpoint`, `ConsumeToActive`.
- **Measured:**
  - **[`bench/agenttest.py`](bench/agenttest.py)**, an agent loop with 20 tools and simulated results ([results](results/runs/agent-2026-10-26.jsonl)):
    - production at 11:17 and the diagnostic build at 12:44: the engine exited at the same step both times (task 2, step 4, 10 messages);
    - with patch 09 and the new template, 3 runs × 4 tasks: 93 steps, 89 tool calls, none returned as text, no error.
  - **[`bench/textcalltest.py`](bench/textcalltest.py)**, three histories in which the assistant wrote text with its calls (one with parallel calls), four seeds each: 12 of 12 replies were clean tool calls, with no markup or turn tokens in the text.
  - **[`config/templates/tplcases.py`](config/templates/tplcases.py):** the agent template renders exactly as Google's for plain chat, multi-turn chat, call-only turns, parallel calls and final answers; only the two text-with-calls shapes differ.
  - **Not measured:**
    - the old template on the same text-with-calls histories, which would have meant another restart;
    - speed, since patch 09 changes one admission count and the template changes only how those histories render.
- **Also:** tonight's test builds `apps-final5` and `apps-final6` were rebuilt on `e9203c5`, so the night-2 runs carry the fix.
- **Rollback:** restore `/etc/systemd/system/gemma4-31b-{fast,262k}.service.bak-20261026-114937` (`apps-final4`, artifact template), then `systemctl daemon-reload` and [`bench/deploy_fast.sh`](bench/deploy_fast.sh).

## 2026-10-26 11:08: web UI tool calling switched on

- **Changed, UI:** the global switch under Settings → Tools is on (`tools_enabled` in the UI's database, set with [`config/ninfer-ui/tools-on.mjs`](config/ninfer-ui/tools-on.mjs)). UI chats now get the built-in `current_time` and `calculator`. No restart; a page reload shows "Use tools" in the Sampling drawer.
- **Why:** 0.1.0 ships with tools off, so UI chats never offered the model a tool.
- **Not changed, engine:** 8080 already served tool calls. A `get_weather` test returned `finish_reason: tool_calls` with `{"city": "Copenhagen"}`, streamed and not, so agents calling 8080 directly were never affected.
- **Rollback:** switch it off as admin under Settings → Tools.

## 2026-10-26 07:39: request counters on `/metrics`, and web UI 0.1.0 (every client on the Monitor)

- **Changed, engine:** `gemma4-31b-fast` and `gemma4-31b-262k` now run `build/apps-final4`: NInferno `6ff9931`, which is `1a3d564` plus [patch 08](patches/ninferno/). Same flags; the fast unit restarted at a moment with no request in flight, and 8080 was down for 36 s ([`bench/deploy_final4.sh`](bench/deploy_final4.sh)).
- **Changed, UI:** `ninfer-ui` runs release 0.1.0 (commit `1492c70`), ported from iNordic's vLLM fork of the same UI. The Monitor reads the engine's `/metrics` once a second, so it shows every client (agents, eval runners, the UI itself), not only the UI's own requests:
  - server activity (running, waiting, reading or generating), with live Speed and a new Prefill card;
  - recent requests with a Source column (UI or API), with the server's own prefill and decode speeds;
  - a totals line: requests, cached prompt tokens, prefix-cache hit rate, MTP acceptance.

  Also ported: one row of cards per GPU, chat and thinking-box scrolling fixes, and lighter rendering of long thinking. `NINFER_ENGINE_METRICS=off` turns the polling off.
- **Patch 08:** `/metrics` adds, over completed requests since start:
  - their count and summed seconds (e2e, queue, TTFT, prefill, decode);
  - their prompt, cached and output tokens;
  - speculative rounds, drafted and accepted tokens;
  - `ninferno_requests_finished_total{finish_reason}`.

  A scraper differencing two samples gets each request that finished alone in between exactly.
- **Why:** the old Monitor never saw traffic sent straight to 8080. This box's main client (an agent) and the eval runner both do that. Also, `ninferno_reused_prompt_tokens_total` stays at 0 for an agent turn resumed from cache; the per-request cached count is that request's own context-cache hit.
- **Measured:** no speed change; `apps-final4` is `1a3d564` plus counter bookkeeping.
  - **Live test:** on the new build behind a 0.1.0 test UI ([`bench/mtest_start.sh`](bench/mtest_start.sh), [log](results/runs/mtest.log), [rerun](results/runs/mtest_rerun.txt)), a UI chat was tagged UI, and a direct API request and an API tool call were tagged API, each with prefill and decode speed.
  - **Cached agent turns:** 16,495 of 17,449 prompt tokens cached on one turn, an 84.1% prefix-hit rate overall, and MTP acceptance 3.8 tokens per step.
  - **Tests:** NInferno's HTTP-routes test (with a new check for the counters) and the UI's 69 tests pass.
- **Rollback:**
  - **Engine:** restore `/etc/systemd/system/gemma4-31b-{fast,262k}.service.bak-20261026-063712` (`apps-final3`), then `systemctl daemon-reload` and [`bench/deploy_fast.sh`](bench/deploy_fast.sh).
  - **UI:** `mv /opt/ninfer-ui/app /opt/ninfer-ui/app-0.1.0 && mv /opt/ninfer-ui/previous-20261026-063903 /opt/ninfer-ui/app && systemctl restart ninfer-ui`. The 0.0.9 Monitor ignores the new counters.

## 2026-10-26: documentation published

- This repository: build route, configs, patches 01–07, benchmarks, every result and run log, and the reports.
- Unit descriptions corrected from "K8" to "K7" to match `--draft-tokens 7` (daemon-reload, no restart).

## 2026-10-26 06:10: decode candidate in production (night 1)

- **Changed:** `gemma4-31b-fast` and `gemma4-31b-262k` now run `build/apps-final3` (NInferno `1a3d564`, adding patches 06 and 07) with `--draft-tokens 7` instead of 8. Applied with [`bench/apply_candidate.sh`](bench/apply_candidate.sh).
- **Why:** with 8 drafts the verify runs 9 tokens and lands in the slower 16-token kernel bucket. With 7 it stays in the 8-token bucket, where patch 06 (QKV schedule) and patch 07 (Q6 head on the 8-column tile) also speed it up.
- **Measured, live:** chat decode 118.4 / 114.0 tok/s against 108.6 / 105.6; prefill 4,154 tok/s at 8K. Quality the same: GSM8K 198, MMLU-Pro 147 (inside the 147–152 band), HumanEval 161. Needles 9/9 at 32K, 67K and 130K; the 262K profile 2/2 at 185K and 231K ([night-1 report](research/2026-10-26-night1-decode.md)).
- **Rollback:** [`bench/revert_candidate.sh`](bench/revert_candidate.sh) restores the unit-file backups `*.bak-20261026-051034` (`apps-final`, 8 drafts).
- **Tested and not adopted:** W4A4 verify; shortlist drafter heads (64K / 96K rows); 16-token schedule variants; SIMT head kernels at 7–8 tokens; Q4_0 prefill on the W4A8 tile (+10% speed, +2% perplexity); 6, 12 and 15 drafts.

## 2026-10-26 00:05: NInfer web UI installed

- **Changed:** `ninfer-ui.service` (release 0.0.9, commit `ef4ebaa`) on ports 8081 (stats) and 8443 (sign-in), upstream `127.0.0.1:8080`, its own self-signed certificate and accounts database. Enabled at boot. Files: [`config/ninfer-ui/`](config/ninfer-ui/).
- **Rollback:** `systemctl disable --now ninfer-ui`.

## 2026-10-25 23:20: day-1 build in production

- **Changed:** the abliterated Q4_K (`gemma4-31b-abl`, plain decode, 32K) replaced by two units on NInferno `4e58bc6` (patches 01–05):
  - `gemma4-31b-fast`: QAT-NVFP4, QAT assistant with 8 drafts, K-only FP8 KV, FP8 attention, GeGLU split, 155K at chunk 1024; on 8080.
  - `gemma4-31b-262k`: QAT-UD Q4, same drafter and KV, 262,144 at chunk 512; installed, stopped.
- **Measured:** decode 33 → 108.6 tok/s on chat; prefill 1.4K → 4.2K at 8K; context 32K → 155K / 262K. MMLU-Pro 150–152 on NVFP4 against 161 on the abliterated Q4_K and 164–165 on QAT Q4 ([day-1 report](research/2026-10-25-day1-results.md)).
- **Rollback:** `systemctl start gemma4-31b-abl` (that unit is still installed).
