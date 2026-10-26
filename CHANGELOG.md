# Changelog

Every change to the running setup, newest first. Each entry names what changed on the box, why, the measurement behind it, and how to roll it back. The README, BUILD.md, configs and results are updated in the same commit. Times are CET.

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
