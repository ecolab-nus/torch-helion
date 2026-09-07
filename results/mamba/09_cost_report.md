# Cost report (mamba)

Hardware `2d_mesh_torus.mlir`: 8x8 mesh, L1 1,398,784 bytes/core.
Estimated end-to-end time: **727,068 cycles** (analytic model, incl. 2,000 cycles per kernel boundary).

| kernel | kind | tiles | est. cycles | waves | iters | load/iter | compute/iter | epilogue | store | L1 bytes | frontend | exploration | Loom cycles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k0_gemm | gemm | {'tile_t': 128, 'tile_n': 64, 'tile_k': 256} | 54,863 | 1 | 2 | 13,921 | 19,908 | 0 | 1,126 | 327,680 | ok | ok | 56,320 |
| k1_gemm | gemm | {'tile_t': 64, 'tile_n': 128, 'tile_k': 256} | 30,808 | 1 | 2 | 10,081 | 9,954 | 0 | 563 | 245,760 | ok | ok | 30,016 |
| k2_gemm | gemm | {'tile_t': 32, 'tile_n': 32, 'tile_k': 256} | 30,849 | 1 | 2 | 3,910 | 12,758 | 956 | 468 | 141,312 | ok | ok | 12,928 |
| k3_gemm | gemm | {'tile_t': 128, 'tile_n': 64, 'tile_k': 32} | 13,426 | 1 | 1 | 2,643 | 9,656 | 0 | 1,126 | 98,304 | ok | ok | 13,632 |
| k4_gemm | gemm | {'tile_t': 64, 'tile_n': 128, 'tile_k': 256} | 30,808 | 1 | 2 | 10,081 | 9,954 | 0 | 563 | 245,760 | ok | ok | 30,016 |
| k5_gemm | gemm | {'tile_t': 64, 'tile_n': 128, 'tile_k': 256} | 54,490 | 1 | 2 | 10,081 | 16,870 | 8,416 | 2,253 | 397,312 | ok | ok | 64,064 |
| k6_ssd | ssd | {'tile_m': 256, 'tile_n': 256} | 66,883 | 1 | 1 | 7,714 | 48,824 | 5,859 | 563 | 311,296 | ok | ok | 146,944 |
| k7_gemm | gemm | {'tile_t': 32, 'tile_n': 128, 'tile_k': 256} | 67,236 | 1 | 4 | 8,334 | 14,294 | 1,218 | 509 | 264,192 | ok | ok | 41,088 |
| k8_gemm | gemm | {'tile_t': 32, 'tile_n': 32, 'tile_k': 256} | 30,849 | 1 | 2 | 3,910 | 12,758 | 956 | 468 | 141,312 | ok | ok | 12,928 |
| k9_gemm | gemm | {'tile_t': 128, 'tile_n': 64, 'tile_k': 32} | 13,426 | 1 | 1 | 2,643 | 9,656 | 0 | 1,126 | 98,304 | ok | ok | 13,632 |
| k10_gemm | gemm | {'tile_t': 64, 'tile_n': 128, 'tile_k': 256} | 30,808 | 1 | 2 | 10,081 | 9,954 | 0 | 563 | 245,760 | ok | ok | 30,016 |
| k11_gemm | gemm | {'tile_t': 128, 'tile_n': 64, 'tile_k': 256} | 120,991 | 1 | 2 | 20,507 | 47,788 | 2,655 | 2,253 | 663,552 | ok | ok | 114,240 |
| k12_ssd | ssd | {'tile_m': 256, 'tile_n': 256} | 66,883 | 1 | 1 | 7,714 | 48,824 | 5,859 | 563 | 311,296 | ok | ok | 146,944 |
| k13_gemm | gemm | {'tile_t': 32, 'tile_n': 128, 'tile_k': 256} | 67,236 | 1 | 4 | 8,334 | 14,294 | 1,218 | 509 | 264,192 | ok | ok | 41,088 |
| k14_gemm | gemm | {'tile_t': 32, 'tile_n': 128, 'tile_k': 256} | 17,512 | 1 | 2 | 1,819 | 6,388 | 2,409 | 509 | 133,120 | ok | ok | 17,984 |

Kernel-to-kernel links: 23 intermediate tensors, 21,037,056 bytes through DRAM.

- `matmul_134` float16[512, 1024]: k9_gemm -> k12_ssd
- `mul_92` float16[512, 1024]: k11_gemm -> k12_ssd
- `where_1` float16[512, 32]: k8_gemm -> k9_gemm
- `mul_22` float16[512, 1024]: k11_gemm -> k12_ssd
- `mul_54` float16[512, 1024]: k5_gemm -> k6_ssd
- `matmul_52` float16[512, 1024]: k1_gemm -> k5_gemm
- `mul_50` float16[512, 1024]: k5_gemm -> k6_ssd
- `mul_96` float16[512, 1024]: k11_gemm -> k12_ssd
- `matmul_135` float16[512, 1024]: k9_gemm -> k10_gemm
- `mul_100` float16[512, 1024]: k11_gemm -> k12_ssd
- `matmul_36` float16[512, 1024]: k0_gemm -> k5_gemm
- `matmul_124` float16[512, 1024]: k3_gemm -> k4_gemm
- `add_4` float16[512, 512]: k7_gemm -> k8_gemm, k11_gemm, k13_gemm
- `mul_9` float16[512, 1024]: k5_gemm -> k6_ssd
- `where` float16[512, 32]: k2_gemm -> k3_gemm
- `matmul_137` float16[512, 1024]: k10_gemm -> k12_ssd
- `mul_58` float16[512, 1024]: k5_gemm -> k6_ssd
- `mul_10` float16[512, 1024]: k6_ssd -> k7_gemm
- `matmul_48` float16[512, 1024]: k0_gemm -> k5_gemm
- `add_9` float16[512, 512]: k13_gemm -> k14_gemm
- `matmul_126` float16[512, 1024]: k4_gemm -> k6_ssd
- `matmul_123` float16[512, 1024]: k3_gemm -> k6_ssd
- `mul_23` float16[512, 1024]: k12_ssd -> k13_gemm
