#!/bin/sh
cd /opt/models/dl
curl -sL -C - --fail --retry 10 --retry-delay 10 -o gemma-4-31B-it-qat-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/resolve/main/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf && echo QATUD_DONE
curl -sL -C - --fail --retry 10 --retry-delay 10 -o abl-bf16.gguf https://huggingface.co/huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-GGUF/resolve/main/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-bf16.gguf && echo ABLBF16_DONE
