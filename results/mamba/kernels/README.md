# Generated kernels

Program `mamba`: 15 Helion kernels in execution order.

## k0_gemm (gemm)

Inputs: `lhs` ← `u` float16[512, 512], `w0` ← `permute_cols_1_norm_35` float16[512, 1024], `w1` ← `permute_cols_2_cols_4_norm_47` float16[512, 1024]
Outputs: `matmul_36` float16[512, 1024], `matmul_48` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k0_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k0_gemm.json --njobs 8 --debug --topk-candidates 1

## k1_gemm (gemm)

Inputs: `lhs` ← `u` float16[512, 512], `w0` ← `permute_cols_2_cols_5_norm_51` float16[512, 1024]
Outputs: `matmul_52` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k1_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k1_gemm.json --njobs 8 --debug --topk-candidates 1

## k2_gemm (gemm)

Inputs: `lhs` ← `u` float16[512, 512], `w0` ← `permute_cols_3_norm_43` float16[512, 32], `in0` ← `layers_0_mixer_dt_bias_full_148` float16[512, 32]
Outputs: `where` float16[512, 32]

    uv run python /workspace/torch-helion/results/mamba/kernels/k2_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k2_gemm.json --njobs 8 --debug --topk-candidates 1

## k3_gemm (gemm)

Inputs: `lhs` ← `where` float16[512, 32], `w0` ← `ssd_expand_A_122` float32[32, 1024], `w1` ← `ssd_expand_121` float16[32, 1024]
Outputs: `matmul_124` float16[512, 1024], `matmul_123` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k3_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k3_gemm.json --njobs 8 --debug --topk-candidates 1

## k4_gemm (gemm)

Inputs: `lhs` ← `ssd_decay_125` float16[512, 512], `w0` ← `matmul_124` float16[512, 1024]
Outputs: `matmul_126` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k4_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k4_gemm.json --njobs 8 --debug --topk-candidates 1

## k5_gemm (gemm)

Inputs: `lhs` ← `u` float16[512, 512], `w0` ← `permute_cols_2_cols_6_norm_55` float16[512, 1024], `in0` ← `matmul_36` float16[512, 1024], `in1` ← `matmul_48` float16[512, 1024], `in2` ← `matmul_52` float16[512, 1024]
Outputs: `mul_50` float16[512, 1024], `mul_54` float16[512, 1024], `mul_58` float16[512, 1024], `mul_9` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k5_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k5_gemm.json --njobs 8 --debug --topk-candidates 1

## k6_ssd (ssd)

Inputs: `c` ← `mul_58` float16[512, 1024] viewed as [2, 256, 32, 32], `b` ← `mul_54` float16[512, 1024] viewed as [2, 256, 32, 32], `x` ← `mul_50` float16[512, 1024] viewed as [2, 256, 32, 32], `cum` ← `matmul_126` float16[512, 1024] viewed as [2, 256, 32, 32], `dt` ← `matmul_123` float16[512, 1024] viewed as [2, 256, 32, 32], `mask` ← `ssd_causal_127` float16[256, 256], `dskip` ← `ssd_D_128` float16[512, 1024] viewed as [2, 256, 32, 32], `xskip` ← `mul_50` float16[512, 1024] viewed as [2, 256, 32, 32], `in0` ← `mul_9` float16[512, 1024] viewed as [2, 256, 32, 32]
Outputs: `mul_10` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k6_ssd.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k6_ssd.json --njobs 8 --debug --topk-candidates 1

## k7_gemm (gemm)

Inputs: `lhs` ← `mul_10` float16[512, 1024], `w0` ← `permute_11_norm_64` float16[1024, 512], `in0` ← `u` float16[512, 512]
Outputs: `add_4` float16[512, 512]

    uv run python /workspace/torch-helion/results/mamba/kernels/k7_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k7_gemm.json --njobs 8 --debug --topk-candidates 1

## k8_gemm (gemm)

Inputs: `lhs` ← `add_4` float16[512, 512], `w0` ← `permute_12_cols_9_norm_85` float16[512, 32], `in0` ← `layers_1_mixer_dt_bias_full_149` float16[512, 32]
Outputs: `where_1` float16[512, 32]

    uv run python /workspace/torch-helion/results/mamba/kernels/k8_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k8_gemm.json --njobs 8 --debug --topk-candidates 1

## k9_gemm (gemm)

Inputs: `lhs` ← `where_1` float16[512, 32], `w0` ← `ssd_expand_A_133` float32[32, 1024], `w1` ← `ssd_expand_121` float16[32, 1024]
Outputs: `matmul_135` float16[512, 1024], `matmul_134` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k9_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k9_gemm.json --njobs 8 --debug --topk-candidates 1

## k10_gemm (gemm)

Inputs: `lhs` ← `ssd_decay_125` float16[512, 512], `w0` ← `matmul_135` float16[512, 1024]
Outputs: `matmul_137` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k10_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k10_gemm.json --njobs 8 --debug --topk-candidates 1

## k11_gemm (gemm)

Inputs: `lhs` ← `add_4` float16[512, 512], `w0` ← `permute_12_cols_7_norm_77` float16[512, 1024], `w1` ← `permute_12_cols_8_cols_10_norm_89` float16[512, 1024], `w2` ← `permute_12_cols_8_cols_11_norm_93` float16[512, 1024], `w3` ← `permute_12_cols_8_cols_12_norm_97` float16[512, 1024]
Outputs: `mul_92` float16[512, 1024], `mul_96` float16[512, 1024], `mul_100` float16[512, 1024], `mul_22` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k11_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k11_gemm.json --njobs 8 --debug --topk-candidates 1

## k12_ssd (ssd)

Inputs: `c` ← `mul_100` float16[512, 1024] viewed as [2, 256, 32, 32], `b` ← `mul_96` float16[512, 1024] viewed as [2, 256, 32, 32], `x` ← `mul_92` float16[512, 1024] viewed as [2, 256, 32, 32], `cum` ← `matmul_137` float16[512, 1024] viewed as [2, 256, 32, 32], `dt` ← `matmul_134` float16[512, 1024] viewed as [2, 256, 32, 32], `mask` ← `ssd_causal_127` float16[256, 256], `dskip` ← `ssd_D_128` float16[512, 1024] viewed as [2, 256, 32, 32], `xskip` ← `mul_92` float16[512, 1024] viewed as [2, 256, 32, 32], `in0` ← `mul_22` float16[512, 1024] viewed as [2, 256, 32, 32]
Outputs: `mul_23` float16[512, 1024]

    uv run python /workspace/torch-helion/results/mamba/kernels/k12_ssd.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k12_ssd.json --njobs 8 --debug --topk-candidates 1

## k13_gemm (gemm)

Inputs: `lhs` ← `mul_23` float16[512, 1024], `w0` ← `permute_23_norm_106` float16[1024, 512], `in0` ← `add_4` float16[512, 512]
Outputs: `add_9` float16[512, 512]

    uv run python /workspace/torch-helion/results/mamba/kernels/k13_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k13_gemm.json --njobs 8 --debug --topk-candidates 1

## k14_gemm (gemm)

Inputs: `lhs` ← `add_9` float16[512, 512], `in0` ← `add_9` float16[512, 512], `in1` ← `norm_f_weight_full_117` float16[512, 512]
Outputs: `mul_118` float16[512, 512]

    uv run python /workspace/torch-helion/results/mamba/kernels/k14_gemm.py --config /workspace/torch-helion/results/mamba/kernels/config_files/k14_gemm.json --njobs 8 --debug --topk-candidates 1
