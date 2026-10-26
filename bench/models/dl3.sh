#!/bin/sh
cd /opt/models/dl2
g() { curl -sL -C - --fail --retry 10 --retry-delay 10 -o "$2" "https://huggingface.co/$1/resolve/main/$3" && echo "DONE $2"; }
for f in config.json generation_config.json tokenizer_config.json model.safetensors tokenizer.json; do g google/gemma-4-31B-it-qat-q4_0-unquantized-assistant qat-assistant/$f $f; done
g unsloth/gemma-4-31B-it-qat-GGUF mtp-qat-Q8_0.gguf MTP/mtp-gemma-4-31B-it-Q8_0.gguf
g unsloth/gemma-4-31B-it-qat-GGUF mtp-qat-Q4_0.gguf MTP/mtp-gemma-4-31B-it-Q4_0.gguf
g Anbeeld/gemma-4-31B-it-DFlash-GGUF dflash-Q4_K_M.gguf gemma4-31b-it-dflash-Q4_K_M.gguf
g Pajari/gemma-4-31B-it-qat-Q4_0_V2 pajari-Q4_0_V2.gguf gemma-4-31B-it-qat-gguf-model-Q4_0_V2_QAT.gguf
g Pajari/gemma-4-31B-it-qat-Q4_0_V2 pajari-mtp-Q4_0_V2.gguf gemma-4-31B-it-qat-assistant-gguf-model-MTP-Q4_0_V2_QAT.gguf
g llmfan46/gemma-4-31B-it-qat-q4_0-uncensored-heretic-GGUF heretic-qat-Q4_0.gguf gemma-4-31B-it-qat-q4_0-uncensored-heretic-Q4_0.gguf
echo ALL_DONE
