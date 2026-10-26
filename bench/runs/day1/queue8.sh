#!/bin/bash
# queue8: after queue7 - Nsight profile of prefill at 32K and 128K (kernel time split)
cd /root/gbench
until grep -q "QUEUE7 DONE" queue7.log; do sleep 15; done
./srv.sh stop; sleep 2
QU=/opt/models/dl/.ninferno-cache/gemma-4-31B-it-qat-UD-Q4_K_XL.gemma4_published.ninferno
G=/opt/NInferno/build/apps/ninferno-gemma4
for n in 32768 131072; do
    echo "=== $(date +%T) profile prefill $n"
    nsys profile -f true -o /root/gbench/prof_$n --trace=cuda --sample=none --cpuctxsw=none \
        $G $QU bench --tokens /root/gbench/prompt128k.i32 --prompt $n --decode 16 --kv-dtype fp8 \
        --context $((n + 256)) --chunk 1024 > /root/gbench/prof_$n.out 2>&1
    tail -3 /root/gbench/prof_$n.out
    nsys stats --report cuda_gpu_kern_sum --format csv -o /root/gbench/prof_$n \
        /root/gbench/prof_$n.nsys-rep > /dev/null 2>&1
    python3 /root/gbench/kernsum.py $n
done

echo "=== $(date +%T) ncu attention + gemm kernels at 32K"
/usr/local/cuda-13.2/bin/ncu --kernel-name regex:"gemma4_attention_kernel" --launch-skip 400 --launch-count 4 --section SpeedOfLight --section ComputeWorkloadAnalysis --section WarpStateStats --section InstructionStats $G $QU bench --tokens /root/gbench/prompt128k.i32 --prompt 32768 --decode 4 --kv-dtype fp8 --context 33000 --chunk 1024 > /root/gbench/ncu_attn.txt 2>&1
grep -E "gemma4_attention_kernel|Duration|Compute \(SM\) Throughput|Memory Throughput|Issue Slots Busy|Executed Ipc|Pipe|Warp Cycles Per Issued|Stall" /root/gbench/ncu_attn.txt | head -40
echo "=== QUEUE8 DONE"
