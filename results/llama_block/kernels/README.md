# Generated kernels

Program `llama_block`: 5 Helion kernels in execution order.

## k0_gemm (gemm)

Inputs: `lhs` ← `x` float16[512, 256], `w0` ← `permute_norm_40` float16[256, 256], `w1` ← `permute_2_norm_48` float16[256, 256], `w2` ← `permute_4_norm_56` float16[256, 256], `w3` ← `permute_rot_3_norm_44` float16[256, 256], `w4` ← `permute_2_rot_28_norm_52` float16[256, 256], `in0` ← `rope_cos_1` float16[512, 256], `in1` ← `rope_sin_2` float16[512, 256]
Outputs: `mul_59` float16[512, 256], `add_13` float16[512, 256], `add_32` float16[512, 256]

    uv run python /workspace/torch-helion/results/llama_block/kernels/k0_gemm.py --config /workspace/torch-helion/results/llama_block/kernels/config_files/k0_gemm.json --njobs 8 --debug --topk-candidates 1

## k1_attention (attention)

Inputs: `q` ← `add_13` float16[512, 256] viewed as [2, 256, 4, 64], `k` ← `add_32` float16[512, 256] viewed as [2, 256, 4, 64], `v` ← `mul_59` float16[512, 256] viewed as [2, 256, 4, 64]
Outputs: `permute_7` float16[512, 256]

    uv run python /workspace/torch-helion/results/llama_block/kernels/k1_attention.py --config /workspace/torch-helion/results/llama_block/kernels/config_files/k1_attention.json --njobs 8 --debug --topk-candidates 1

## k2_gemm (gemm)

Inputs: `lhs` ← `permute_7` float16[512, 256], `w0` ← `permute_8` float16[256, 256], `in0` ← `x` float16[512, 256]
Outputs: `add_3` float16[512, 256]

    uv run python /workspace/torch-helion/results/llama_block/kernels/k2_gemm.py --config /workspace/torch-helion/results/llama_block/kernels/config_files/k2_gemm.json --njobs 8 --debug --topk-candidates 1

## k3_gemm (gemm)

Inputs: `lhs` ← `add_3` float16[512, 256], `w0` ← `permute_9_norm_65` float16[256, 512], `w1` ← `permute_10_norm_69` float16[256, 512]
Outputs: `mul_10` float16[512, 512]

    uv run python /workspace/torch-helion/results/llama_block/kernels/k3_gemm.py --config /workspace/torch-helion/results/llama_block/kernels/config_files/k3_gemm.json --njobs 8 --debug --topk-candidates 1

## k4_gemm (gemm)

Inputs: `lhs` ← `mul_10` float16[512, 512], `w0` ← `permute_11` float16[512, 256], `in0` ← `add_3` float16[512, 256]
Outputs: `add_5` float16[512, 256]

    uv run python /workspace/torch-helion/results/llama_block/kernels/k4_gemm.py --config /workspace/torch-helion/results/llama_block/kernels/config_files/k4_gemm.json --njobs 8 --debug --topk-candidates 1
