"""n5_skipcount.py <tree>: a counting build, never for production. For each warp's score tile of the
D = 512 FP8 kernel, counts how far the tile's best score sits below the running maximum, in log2
units: the share of tiles skip-softmax could drop at a threshold of 2^-6 .. 2^-14."""
import sys

root = sys.argv[1]
k = f"{root}/src/ops/softmax_attention/gemma4/kernel.cuh"
s = open(k).read()
head = "template <class Geo, int WR, int Bc, bool Partial, bool Fp8 = false, bool Chain = false>\n__launch_bounds__(Gemma4Tile<Geo, WR, Bc, Fp8>::Threads, 1) __global__"
assert s.count(head) == 1
s = s.replace(head, "#define NINFERNO_GEMMA4_SKIP_COUNT 1\nstatic __device__ unsigned long long g_skip_count[8];\n\n" + head)
old = "        const float nm1    = fmaxf(m_run[1], bm1);\n"
assert s.count(old) == 1
s = s.replace(old, old + """        if constexpr (Fp8 && D >= 512) {
            const float g0 = nm0 == NegInf ? NegInf : (bm0 - nm0) * scale_log2;
            const float g1 = nm1 == NegInf ? NegInf : (bm1 - nm1) * scale_log2;
            float gap      = fmaxf(g0, g1);
            gap            = fmaxf(gap, __shfl_xor_sync(0xffffffffu, gap, 4));
            gap            = fmaxf(gap, __shfl_xor_sync(0xffffffffu, gap, 8));
            gap            = fmaxf(gap, __shfl_xor_sync(0xffffffffu, gap, 16));
            if ((threadIdx.x & 31) == 0) {
                atomicAdd(&g_skip_count[0], 1ULL);
                if (gap < -6.0f) { atomicAdd(&g_skip_count[1], 1ULL); }
                if (gap < -8.0f) { atomicAdd(&g_skip_count[2], 1ULL); }
                if (gap < -10.0f) { atomicAdd(&g_skip_count[3], 1ULL); }
                if (gap < -12.0f) { atomicAdd(&g_skip_count[4], 1ULL); }
                if (gap < -14.0f) { atomicAdd(&g_skip_count[5], 1ULL); }
            }
        }
""")
open(k, "w").write(s)
l = f"{root}/src/ops/softmax_attention/gemma4/launch.cu"
s = open(l).read()
old = "        kernel<<<grid, Tile::Threads, Tile::SmemBytes, stream>>>(args);\n        CUDA_CHECK(cudaGetLastError());\n    };\n    if constexpr (Bf16Only) {"
assert s.count(old) == 1
s = s.replace(old, """        kernel<<<grid, Tile::Threads, Tile::SmemBytes, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
        if constexpr (Fp8 && Geo::D >= 512) {
            static unsigned long long launches = 0;
            cudaStreamCaptureStatus capture    = cudaStreamCaptureStatusNone;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
            if (capture == cudaStreamCaptureStatusNone && ++launches % 40 == 0) {
                unsigned long long c[8] = {};
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaMemcpyFromSymbol(c, g_skip_count, sizeof(c)));
                std::fprintf(stderr, "ninferno: skipcount %llu %llu %llu %llu %llu %llu\\n", c[0], c[1],
                             c[2], c[3], c[4], c[5]);
            }
        }
    };
    if constexpr (Bf16Only) {""")
first = s.index("#include <")
s = s[:first] + "#include <cstdio>\n" + s[first:]
open(l, "w").write(s)
print("skip counter applied")
