# Config

## `systemd/`

| Unit | Model | Context | Notes |
|---|---|---|---|
| [`gemma4-31b-fast.service`](systemd/gemma4-31b-fast.service) | QAT-NVFP4 (`qat-nvfp4.r31.ninferno`) | 155,000, chunk 1024 | production on port 8080 |
| [`gemma4-31b-262k.service`](systemd/gemma4-31b-262k.service) | QAT-UD Q4 | 262,144, chunk 512 | the max-context and quality option |

Both units:
- run `build/apps-final3/ninferno-serve` (NInferno `1a3d564`) with MTP, 7 drafts, the QAT assistant and FP8 KV;
- set `NINFERNO_GEMMA4_SHARED_VALUES=2` (K-only KV) and `NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1`;
- conflict with each other, so starting one stops the other;
- are not enabled at boot.

## `ninfer-ui/`

The service files for iNordic's NInfer web UI, release 0.0.9 (not included here):
- [`ninfer-ui.service`](ninfer-ui/ninfer-ui.service): a sandboxed `DynamicUser` service running the UI's bundled Node runtime.
- [`ninfer-ui.service.d/accounts.conf`](ninfer-ui/ninfer-ui.service.d/accounts.conf): the state directory for the accounts database, the TLS certificate as systemd credentials, the upstream (`http://127.0.0.1:8080`), and ports 8081 (stats) and 8443 (sign-in). Set `NINFER_UI_HOST` to your host's IP.
- [`ninfer-ui.service.d/monitoring.conf`](ninfer-ui/ninfer-ui.service.d/monitoring.conf): read-only access to the GPU device nodes for the Monitor's `nvidia-smi` stats.

The UI's release-check broker (`ninfer-ui-updater`) is not installed on this box.
