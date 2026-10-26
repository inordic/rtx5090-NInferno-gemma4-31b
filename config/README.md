# Config

## `systemd/`

| Unit | Model | Context | Notes |
|---|---|---|---|
| [`gemma4-31b-fast.service`](systemd/gemma4-31b-fast.service) | QAT-NVFP4 (`qat-nvfp4.r31.ninferno`) | 155,000, chunk 1024 | production on port 8080 |
| [`gemma4-31b-262k.service`](systemd/gemma4-31b-262k.service) | QAT-UD Q4 | 262,144, chunk 512 | the max-context and quality option |

Both units:
- run `build/apps-final7/ninferno-serve` (NInferno `e9203c5`) with MTP, 7 drafts, the QAT assistant and FP8 KV;
- render prompts with [`templates/gemma4-agent.jinja`](templates/gemma4-agent.jinja) (`--chat-template /opt/models/templates/gemma4-agent.jinja`);
- set `NINFERNO_GEMMA4_SHARED_VALUES=2` (K-only KV) and `NINFERNO_GEMMA4_GEGLU_SPLIT_MIN_T=1`;
- conflict with each other, so starting one stops the other;
- are not enabled at boot.

## `templates/`

The chat template both units serve with. NInferno renders prompts with the template stored in the model artifact unless `--chat-template` names another; our QAT-NVFP4 artifact carries the one from its quant's repo, an older Gemma 4 template.

- [`gemma4-agent.jinja`](templates/gemma4-agent.jinja) is Google's Gemma 4 template of 2026-07-09 ([`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja), "Fixed tool-calling loops, turn closures, and thinking content-ordering") with two changes for agents:
  - an assistant message's text goes before its tool calls, the order the model writes them. The template put it after the tool results, as if it were the answer to them;
  - a turn whose tool results end the conversation stays open, so the model continues inside it. With text in the message, Google's template closed the turn and opened no new one, and the model had to write outside any turn.
- The old template also closed the model's turn after any assistant message with text and calls and then wrote the following calls and results outside a turn, for the rest of the conversation. Google's July template already fixes that.
- [`mkagent.py`](templates/mkagent.py) makes `gemma4-agent.jinja` from Google's file: `python3 mkagent.py chat_template.jinja gemma4-agent.jinja`.
- [`tplrender.py`](templates/tplrender.py) prints an agent conversation as each template renders it, and [`tplcases.py`](templates/tplcases.py) compares two templates over seven conversation shapes: `python3 tplcases.py google.jinja gemma4-agent.jinja` shows only the two text-with-calls shapes differ. Both need `jinja2`.

## `ninfer-ui/`

The service files for iNordic's NInfer web UI, release 0.1.0 (not included here). Its Monitor reads the engine's `/metrics` once a second, so it shows every client of 8080; `NINFER_ENGINE_METRICS=off` in the environment turns that off:
- [`ninfer-ui.service`](ninfer-ui/ninfer-ui.service): a sandboxed `DynamicUser` service running the UI's bundled Node runtime.
- [`ninfer-ui.service.d/accounts.conf`](ninfer-ui/ninfer-ui.service.d/accounts.conf): the state directory for the accounts database, the TLS certificate as systemd credentials, the upstream (`http://127.0.0.1:8080`), and ports 8081 (stats) and 8443 (sign-in). Set `NINFER_UI_HOST` to your host's IP.
- [`ninfer-ui.service.d/monitoring.conf`](ninfer-ui/ninfer-ui.service.d/monitoring.conf): read-only access to the GPU device nodes for the Monitor's `nvidia-smi` stats.

The UI's release-check broker (`ninfer-ui-updater`) is not installed on this box.
