# Cost report (mamba_block)

Hardware `2d_mesh_torus.mlir`: 8x8 mesh, L1 1,398,784 bytes/core.
Estimated end-to-end time: **361,705 cycles** (analytic model, incl. 2,000 cycles per kernel boundary).

| kernel | kind | tiles | est. cycles | waves | iters | load/iter | compute/iter | epilogue | store | L1 bytes | frontend | exploration | Loom cycles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k0_gemm | gemm | {'tile_t': 32, 'tile_n': 32, 'tile_k': 256} | 30,849 | 1 | 2 | 3,910 | 12,758 | 956 | 468 | 141,312 | ok | ok | 12,928 |
| k1_gemm | gemm | {'tile_t': 128, 'tile_n': 64, 'tile_k': 32} | 13,426 | 1 | 1 | 2,643 | 9,656 | 0 | 1,126 | 98,304 | ok | ok | 13,632 |
| k2_gemm | gemm | {'tile_t': 64, 'tile_n': 128, 'tile_k': 256} | 30,808 | 1 | 2 | 10,081 | 9,954 | 0 | 563 | 245,760 | ok | ok | 30,016 |
| k3_gemm | gemm | {'tile_t': 128, 'tile_n': 64, 'tile_k': 256} | 120,991 | 1 | 2 | 20,507 | 47,788 | 2,655 | 2,253 | 663,552 | ok | ok | 114,240 |
| k4_ssd | ssd | {'tile_m': 256, 'tile_n': 256} | 66,883 | 1 | 1 | 7,714 | 48,824 | 5,859 | 563 | 311,296 | ok | ok | 146,944 |
| k5_gemm | gemm | {'tile_t': 32, 'tile_n': 128, 'tile_k': 256} | 67,236 | 1 | 4 | 8,334 | 14,294 | 1,218 | 509 | 264,192 | ok | ok | 41,088 |
| k6_gemm | gemm | {'tile_t': 32, 'tile_n': 128, 'tile_k': 256} | 17,512 | 1 | 2 | 1,819 | 6,388 | 2,409 | 509 | 133,120 | ok | ok | 17,984 |

Kernel-to-kernel links: 10 intermediate tensors, 8,945,664 bytes through DRAM.

- `matmul_76` float16[512, 1024]: k1_gemm -> k4_ssd
- `matmul_79` float16[512, 1024]: k2_gemm -> k4_ssd
- `mul_10` float16[512, 1024]: k4_ssd -> k5_gemm
- `mul_45` float16[512, 1024]: k3_gemm -> k4_ssd
- `add_4` float16[512, 512]: k5_gemm -> k6_gemm
- `matmul_77` float16[512, 1024]: k1_gemm -> k2_gemm
- `mul_53` float16[512, 1024]: k3_gemm -> k4_ssd
- `where` float16[512, 32]: k0_gemm -> k1_gemm
- `mul_9` float16[512, 1024]: k3_gemm -> k4_ssd
- `mul_49` float16[512, 1024]: k3_gemm -> k4_ssd
