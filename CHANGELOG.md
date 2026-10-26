# Changelog

Every change to the running setup, newest first. Each entry names what changed on the box, why, the measurement behind it, and how to roll it back. The README, BUILD.md, configs and results are updated in the same commit. Times are CET.

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
