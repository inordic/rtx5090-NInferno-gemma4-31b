#!/bin/bash
until grep -q NVFP4_DL_DONE /opt/models/dl2/nvfp4.log; do sleep 10; done
cd /opt/models/dl2 && nice -n 10 /opt/NInferno/build/apps/ninferno-convert --model /opt/models/dl2/qat-nvfp4 --out /opt/models/dl2/qat-nvfp4.vq8.ninferno --vocab q8 --components text 2>&1 | tail -4
ls -la /opt/models/dl2/qat-nvfp4.vq8.ninferno && echo NVFP4_CONVERT_DONE
