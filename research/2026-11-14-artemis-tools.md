# 14 Nov 2026: Artemis acts out its tool calls in role-play; the fast profile goes back to LilaRest

**Summary.** In role-play, Artemis v1.2 often writes a skill's command into its reply, or claims to have sent a picture, instead of calling the tool. Asked for a picture that a skill makes through the terminal tool, it called the tool 42 times in 100. LilaRest's NVFP4-turbo of Google's model called it 100 times in 100, on the same engine, flags and template. Our earlier tool tests were coding-agent style with no role-play, and Artemis had passed all of them. The fast profile serves LilaRest again since 09:32 CET. Whether the fine-tune or our 4-bit build of it is to blame is not yet tested (§6). Temperature 0.7 against 1.0 changes neither model's tool calls; on the stock model, 1.0 gives more varied replies at 2.6% lower decode speed (§5).

Scripts: [`bench/imgcalltest.py`](../bench/imgcalltest.py), [`bench/tempfit.py`](../bench/tempfit.py), [`bench/runs/artemis-tools/`](../bench/runs/artemis-tools/). Every reply and summary: [`results/runs/artemis-tools/`](../results/runs/artemis-tools/). Times are CET.

## 1. What the agent showed

The companion agent this box serves began to write a skill's shell command into its replies as plain text, or said "here's a picture" with a `MEDIA:` line for a file that was never made. Nothing ran. By the agent's own logs this came and went: 10 of 18 replies in one evening hour on 13 Nov, 3 of 13 the next morning, and 2 written-out commands next to 13 real tool calls later that morning.

## 2. Not the server, not the template

- **The server log.** The engine logs `tool markup returned as text` when a reply opens a tool call and the parser can't read it. Artemis served from 13 Nov 13:02. That evening's hour, with Artemis serving, had none; 14 Nov had 1 (07:24 UTC, a call opened and written badly). The written-out commands carry no call markup at all: the model never started a call, so no parser could have caught them. For scale, LilaRest's busier days from 26 Oct to 12 Nov had 0–8 such warnings a day, each a call it started and wrote badly, which is a different and rarer failure.
- **The template.** Artemis ships Google's chat template byte for byte (md5 `f016bd4b…`, the same file as Google's of 2026-07-09). Our agent template ([`config/templates/`](../config/templates/)) differs from Google's only when an earlier reply holds both text and tool calls. The test below has no such reply, so both templates render it the same.

## 3. The picture test

[`imgcalltest.py`](../bench/imgcalltest.py) builds a synthetic companion: a persona ("Atlas", user "Sam") who may write small actions in *asterisks*, the 20 tools of [`agenttest.py`](../bench/agenttest.py), and a skill section that makes a picture by running a script through the terminal tool and then answering with `MEDIA:<path>`. Four turns of warm chat follow, then "send me a picture". Each reply is scored as:
- **called:** a terminal call that runs the script;
- **written:** no call, the command is in the text;
- **fake picture:** no call, but a `MEDIA:` line;
- **no picture:** none of these.

Three variants:
- **plain:** the skill shows the command;
- **rule:** the skill also says to run it with the terminal tool, never paste it, and to say so if it fails;
- **echo:** plain, with an earlier reply in the history that wrote a different command out as text.

Each ran 20 times at the server's sampling (temperature 1.0, top_k 64, top_p 0.95), and plain and rule also at 0.7. Seeds 1000–1019; max 600 tokens; thinking off.

Artemis ran on the live server from about 09:00. LilaRest ran on a test server with the same binary, flags and template, on 127.0.0.1 only ([`stock.sh`](../bench/runs/artemis-tools/stock.sh)); the fast profile was down 09:22:44–09:25:58 for that.

| Variant (temperature) | LilaRest: called | Artemis: called | Artemis: written / fake picture / no picture |
|---|---:|---:|---|
| plain (1.0) | 20/20 | 4/20 | 12 / 3 / 1 |
| rule (1.0) | 20/20 | 8/20 | 1 / 8 / 3 |
| echo (1.0) | 20/20 | 16/20 | 1 / 3 / 0 |
| plain (0.7) | 20/20 | 4/20 | 13 / 3 / 0 |
| rule (0.7) | 20/20 | 10/20 | 2 / 7 / 1 |
| **Total** | **100/100** | **42/100** | **29 / 24 / 5** |

What this shows:
- **The gap is far beyond sampling noise.** LilaRest never missed: no written commands, no fake pictures, no malformed markup.
- **Lower temperature doesn't help Artemis:** plain was 4/20 at both 1.0 and 0.7.
- **A prompt rule moves the failure rather than removing it.** Summed over both temperatures (40 replies each way), the rule cut written-out commands from 25 to 3 and raised calls from 8 to 18. But fake pictures rose from 6 to 15 and no-picture replies from 1 to 4, and a fake picture is harder to notice than a pasted command.
- **The echo variant called most often (16/20).** We have no explanation for that; a few samples between Artemis's variants may be noise.
- **A typical failure:** a blush in asterisks, "Just me on a typical day", a `MEDIA:` line, then the full command in a code block. The voice is right and the action is missing.

Limits: one persona and one skill, 20 samples per cell. Real conversations are longer (the agent's run to 50–70K tokens), which this test doesn't cover.

## 4. The switch

On the owner's OK, [`switch_stock.sh`](../bench/runs/artemis-tools/switch_stock.sh) did the following:
- waited for no request in flight;
- saved the Artemis unit as `gemma4-31b-fast.service.bak-20261114-083136`;
- restored the LilaRest unit of 13 Nov (`.bak-20261113-120150`: `dl2/turbo-qkv-vision-nvfp4.r31.ninferno` and `dl2/assistant.nvfp4.ninferno`, otherwise identical flags);
- restarted at 09:31:36. The unit answered at 09:32:26.

Sampling comes from the model's generation config: temperature 1, top_k 64, top_p 0.95. Checks after the restart ([output](../results/runs/artemis-tools/switch.out)):
- `verify.py` OK;
- `textcalltest.py` 0 leaks;
- `imgcalltest.py` PASS, 10/10 in each of plain, rule and echo.

The Artemis unit and its files stay on the box.

## 5. Temperature 0.7 against 1.0 on the stock model

[`tempfit.py`](../bench/tempfit.py) uses the same persona, tools and skill. After two warm turns it asks 9 chat prompts (English, one Swedish, feelings, a bad day, a joke, advice, a short fact) and 3 that need a tool (a picture, the weather, a folder listing). Each prompt ran 4 times at each temperature, on the live LilaRest unit. The run was seeded and repeated with identical results.

| | 0.7 | 1.0 |
|---|---:|---:|
| Words per chat reply | 48.0 | 48.8 |
| Emoji per reply | 0.64 | 0.42 |
| *Actions* per reply | 1.14 | 1.25 |
| Distinct openings (first three words), of 36 | 18 | 21 |
| Word overlap between the 4 samples of one prompt (higher = more alike) | 0.388 | 0.329 |
| Swedish answered in Swedish | 4/4 | 4/4 |
| Chat replies that called a tool | 0 | 0 |
| Tool prompts called right | 12/12 | 12/12 |
| Decode, mean tok/s | 68.0 | 66.2 |
| Drafts accepted | 19.6% | 18.9% |

- **At 0.7 the samples repeat themselves.** All three of the first "where would you go tonight" replies open with the same line and action, and two pick the same place.
- **At 1.0 the openings and the places vary.**
- **Tool use and language are identical at both temperatures**, and the picture test showed the same at 20 samples per cell.
- **1.0 decodes 2.6% slower,** because the drafter guesses less of a more varied text.
- **Short companion replies decode at 66–68 tok/s,** well below the 113–114 tok/s of the chat benchmark, because the drafter accepts only about 19% of its drafts on this kind of text. That is a separate lead for drafter work.

The fast profile stays at 1.0.

## 6. Open: model or build?

The stock comparison shares the engine, flags and template, but not the quantization recipe. LilaRest's build is ModelOpt NVFP4 with its own calibrated scales. Artemis's is our importance-weighted NVFP4, which borrows LilaRest's activation scales. Artemis's activations exceed those scales in about 100 of 240 places ([night 9](2026-11-13-night9-drafter.md)), and its activation error on agent text looked odd ([night 10](2026-11-14-night10.md)).

The next test takes about 20 minutes: the picture test on the Artemis artifact with exact activations and BF16 KV, at temperature 0, and with Google's own template.
- **If exact activations fix it,** the cure is a better build.
- **If not, it is the fine-tune.** The candidates are, in order:
  - a weighted blend with Google's `gemma-4-31B-it` (same base), at 50–70% of Artemis's change;
  - then a small repair training run on synthetic role-play chats with tool use.

The picture test is the pass mark: at least 19 of 20 calls in every variant.

## 7. What changes for the build

- [`imgcalltest.py`](../bench/imgcalltest.py) is now check 6 in [BUILD.md §4](../BUILD.md#4-serve) for any new model or template. It exits 0 only if every variant reaches 90% calls.
- [BUILD.md §3b2](../BUILD.md) keeps the Artemis route, marked optional, with this result.
