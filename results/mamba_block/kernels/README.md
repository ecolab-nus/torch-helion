# Generated kernels

Program `mamba_block`: 7 Helion kernels in execution order.

## k0_gemm (gemm)

Inputs: `lhs` ← `u` float16[512, 512], `w0` ← `permute_cols_3_norm_38` float16[512, 32], `in0` ← `layers_0_mixer_dt_bias_full_87` float16[512, 32]
Outputs: `where` float16[512, 32]

    uv run python /workspace/torch-helion/results/mamba_block/kernels/k0_gemm.py --config /workspace/torch-helion/results/mamba_block/kernels/config_files/k0_gemm.json --njobs 16 --debug --topk-candidates 1

## k1_gemm (gemm)

Inputs: `lhs` ← `where` float16[512, 32], `w0` ← `ssd_expand_A_75` float32[32, 1024], `w1` ← `ssd_expand_74` float16[32, 1024]
Outputs: `matmul_77` float16[512, 1024], `matmul_76` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba_block/kernels/k1_gemm.py --config /workspace/torch-helion/results/mamba_block/kernels/config_files/k1_gemm.json --njobs 16 --debug --topk-candidates 1

## k2_gemm (gemm)

Inputs: `lhs` ← `ssd_decay_78` float16[512, 512], `w0` ← `matmul_77` float16[512, 1024]
Outputs: `matmul_79` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba_block/kernels/k2_gemm.py --config /workspace/torch-helion/results/mamba_block/kernels/config_files/k2_gemm.json --njobs 16 --debug --topk-candidates 1

## k3_gemm (gemm)

Inputs: `lhs` ← `u` float16[512, 512], `w0` ← `permute_cols_1_norm_30` float16[512, 1024], `w1` ← `permute_cols_2_cols_4_norm_42` float16[512, 1024], `w2` ← `permute_cols_2_cols_5_norm_46` float16[512, 1024], `w3` ← `permute_cols_2_cols_6_norm_50` float16[512, 1024]
Outputs: `mul_45` float16[512, 1024], `mul_49` float16[512, 1024], `mul_53` float16[512, 1024], `mul_9` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba_block/kernels/k3_gemm.py --config /workspace/torch-helion/results/mamba_block/kernels/config_files/k3_gemm.json --njobs 16 --debug --topk-candidates 1

## k4_ssd (ssd)

Inputs: `c` ← `mul_53` float16[512, 1024] viewed as [2, 256, 32, 32], `b` ← `mul_49` float16[512, 1024] viewed as [2, 256, 32, 32], `x` ← `mul_45` float16[512, 1024] viewed as [2, 256, 32, 32], `cum` ← `matmul_79` float16[512, 1024] viewed as [2, 256, 32, 32], `dt` ← `matmul_76` float16[512, 1024] viewed as [2, 256, 32, 32], `mask` ← `ssd_causal_80` float16[256, 256], `dskip` ← `ssd_D_81` float16[512, 1024] viewed as [2, 256, 32, 32], `xskip` ← `mul_45` float16[512, 1024] viewed as [2, 256, 32, 32], `in0` ← `mul_9` float16[512, 1024] viewed as [2, 256, 32, 32]
Outputs: `mul_10` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba_block/kernels/k4_ssd.py --config /workspace/torch-helion/results/mamba_block/kernels/config_files/k4_ssd.json --njobs 16 --debug --topk-candidates 1

## k5_gemm (gemm)

Inputs: `lhs` ← `mul_10` float16[512, 1024], `w0` ← `permute_11_norm_59` float16[1024, 512], `in0` ← `u` float16[512, 512]
Outputs: `add_4` float16[512, 512]

    uv run python /workspace/torch-helion/results/mamba_block/kernels/k5_gemm.py --config /workspace/torch-helion/results/mamba_block/kernels/config_files/k5_gemm.json --njobs 16 --debug --topk-candidates 1

## k6_gemm (gemm)

Inputs: `lhs` ← `add_4` float16[512, 512], `in0` ← `add_4` float16[512, 512], `in1` ← `norm_f_weight_full_70` float16[512, 512]
Outputs: `mul_71` float16[512, 512]

    uv run python /workspace/torch-helion/results/mamba_block/kernels/k6_gemm.py --config /workspace/torch-helion/results/mamba_block/kernels/config_files/k6_gemm.json --njobs 16 --debug --topk-candidates 1
