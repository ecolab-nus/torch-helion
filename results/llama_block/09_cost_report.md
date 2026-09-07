# Cost report (llama_block)

Hardware `2d_mesh_torus.mlir`: 8x8 mesh, L1 1,398,784 bytes/core.
Estimated end-to-end time: **158,651 cycles** (analytic model, incl. 2,000 cycles per kernel boundary).

| kernel | kind | tiles | est. cycles | waves | iters | load/iter | compute/iter | epilogue | store | L1 bytes | frontend | exploration | Loom cycles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k0_gemm | gemm | {'tile_t': 64, 'tile_n': 32, 'tile_k': 256} | 57,409 | 1 | 1 | 12,932 | 41,326 | 1,708 | 1,444 | 368,640 | ok | ok | - |
| k1_attention | attention | {'tile_m': 32, 'tile_n': 128} | 23,846 | 1 | 2 | 3,857 | 9,171 | 342 | 481 | 98,560 | ok | ok | - |
| k2_gemm | gemm | {'tile_t': 32, 'tile_n': 64, 'tile_k': 128} | 12,249 | 1 | 2 | 3,036 | 3,953 | 826 | 481 | 65,536 | ok | ok | - |
| k3_gemm | gemm | {'tile_t': 64, 'tile_n': 64, 'tile_k': 256} | 34,991 | 1 | 1 | 10,425 | 22,728 | 1,329 | 509 | 331,776 | ok | ok | - |
| k4_gemm | gemm | {'tile_t': 32, 'tile_n': 64, 'tile_k': 128} | 20,155 | 1 | 4 | 3,036 | 3,953 | 826 | 481 | 65,536 | ok | ok | - |

Kernel-to-kernel links: 6 intermediate tensors, 1,835,008 bytes through DRAM.

- `mul_59` float16[512, 256]: k0_gemm -> k1_attention
- `add_13` float16[512, 256]: k0_gemm -> k1_attention
- `mul_10` float16[512, 512]: k3_gemm -> k4_gemm
- `add_3` float16[512, 256]: k2_gemm -> k3_gemm, k4_gemm
- `permute_7` float16[512, 256]: k1_attention -> k2_gemm
- `add_32` float16[512, 256]: k0_gemm -> k1_attention
