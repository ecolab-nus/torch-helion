# Partition plan (llama_block)

Hardware: 2d_mesh_torus.mlir, mesh (8, 8), L1 1,398,784 bytes
Candidates: 104 (legal: 104), planner mode: exhaustive

| # | kernels | est. cycles | legal | note |
|---|---------|-------------|-------|------|
| 103 | 5 | 158,651 | yes | best |
| 101 | 6 | 164,197 | yes |  |
| 102 | 6 | 164,416 | yes |  |
| 47 | 6 | 165,924 | yes |  |
| 71 | 6 | 165,924 | yes |  |
| 91 | 6 | 165,924 | yes |  |
| 97 | 6 | 165,924 | yes |  |
| 51 | 6 | 166,914 | yes |  |
| 53 | 6 | 166,914 | yes |  |
| 65 | 6 | 166,914 | yes |  |
| 67 | 6 | 166,914 | yes |  |
| 85 | 6 | 166,914 | yes |  |
| 87 | 6 | 166,914 | yes |  |
| 45 | 6 | 167,481 | yes |  |
| 73 | 6 | 167,481 | yes |  |
| 93 | 6 | 167,481 | yes |  |
| 99 | 6 | 167,481 | yes |  |
| 95 | 7 | 169,742 | yes |  |
| 100 | 7 | 169,962 | yes |  |
| 41 | 7 | 170,870 | yes |  |
| 49 | 7 | 170,870 | yes |  |
| 59 | 7 | 170,870 | yes |  |
| 63 | 7 | 170,870 | yes |  |
| 77 | 7 | 170,870 | yes |  |
| 79 | 7 | 170,870 | yes |  |
| 83 | 7 | 170,870 | yes |  |
| 13 | 7 | 171,436 | yes |  |
| 23 | 7 | 171,436 | yes |  |
| 33 | 7 | 171,436 | yes |  |
| 39 | 7 | 171,436 | yes |  |
| 43 | 7 | 171,436 | yes |  |
| 61 | 7 | 171,436 | yes |  |
| 69 | 7 | 171,436 | yes |  |
| 81 | 7 | 171,436 | yes |  |
| 89 | 7 | 171,436 | yes |  |
| 15 | 7 | 171,573 | yes |  |
| 17 | 7 | 171,573 | yes |  |
| 21 | 7 | 171,573 | yes |  |
| 25 | 7 | 171,573 | yes |  |
| 29 | 7 | 171,573 | yes |  |
| 31 | 7 | 171,573 | yes |  |
| 37 | 7 | 171,573 | yes |  |
| 57 | 7 | 171,573 | yes |  |
| 46 | 7 | 171,689 | yes |  |
| 70 | 7 | 171,689 | yes |  |
| 90 | 7 | 171,689 | yes |  |
| 96 | 7 | 171,689 | yes |  |
| 50 | 7 | 172,679 | yes |  |
| 52 | 7 | 172,679 | yes |  |
| 64 | 7 | 172,679 | yes |  |
| 66 | 7 | 172,679 | yes |  |
| 84 | 7 | 172,679 | yes |  |
| 86 | 7 | 172,679 | yes |  |
| 44 | 7 | 173,246 | yes |  |
| 72 | 7 | 173,246 | yes |  |
| 92 | 7 | 173,246 | yes |  |
| 98 | 7 | 173,246 | yes |  |
| 75 | 8 | 174,688 | yes |  |
| 3 | 8 | 175,391 | yes |  |
| 5 | 8 | 175,391 | yes |  |
| 7 | 8 | 175,391 | yes |  |
| 9 | 8 | 175,391 | yes |  |
| 11 | 8 | 175,391 | yes |  |
| 19 | 8 | 175,391 | yes |  |
| 27 | 8 | 175,391 | yes |  |
| 35 | 8 | 175,391 | yes |  |
| 55 | 8 | 175,391 | yes |  |
| 94 | 8 | 175,507 | yes |  |
| 40 | 8 | 176,634 | yes |  |
| 48 | 8 | 176,634 | yes |  |
| 58 | 8 | 176,634 | yes |  |
| 62 | 8 | 176,634 | yes |  |
| 76 | 8 | 176,634 | yes |  |
| 78 | 8 | 176,634 | yes |  |
| 82 | 8 | 176,634 | yes |  |
| 12 | 8 | 177,201 | yes |  |
| 22 | 8 | 177,201 | yes |  |
| 32 | 8 | 177,201 | yes |  |
| 38 | 8 | 177,201 | yes |  |
| 42 | 8 | 177,201 | yes |  |
| 60 | 8 | 177,201 | yes |  |
| 68 | 8 | 177,201 | yes |  |
| 80 | 8 | 177,201 | yes |  |
| 88 | 8 | 177,201 | yes |  |
| 14 | 8 | 177,338 | yes |  |
| 16 | 8 | 177,338 | yes |  |
| 20 | 8 | 177,338 | yes |  |
| 24 | 8 | 177,338 | yes |  |
| 28 | 8 | 177,338 | yes |  |
| 30 | 8 | 177,338 | yes |  |
| 36 | 8 | 177,338 | yes |  |
| 56 | 8 | 177,338 | yes |  |
| 1 | 9 | 179,210 | yes |  |
| 74 | 9 | 180,453 | yes |  |
| 2 | 9 | 181,156 | yes |  |
| 4 | 9 | 181,156 | yes |  |
| 6 | 9 | 181,156 | yes |  |
| 8 | 9 | 181,156 | yes |  |
| 10 | 9 | 181,156 | yes |  |
| 18 | 9 | 181,156 | yes |  |
| 26 | 9 | 181,156 | yes |  |
| 34 | 9 | 181,156 | yes |  |
| 54 | 9 | 181,156 | yes |  |
| 0 | 10 | 184,975 | yes |  |

## Chosen program

```
Program llama_block: 5 kernels, inputs=['x'], outputs=['add_5']
  [gemm] k0_gemm(lhs<-x, w0<-permute_norm_40, w1<-permute_2_norm_48, w2<-permute_4_norm_56, w3<-permute_rot_3_norm_44, w4<-permute_2_rot_28_norm_52, in0<-rope_cos_1, in1<-rope_sin_2) -> mul_59, add_13, add_32 gemms=5 sumsq=True epilogue=19 est=57,409cyc tiles={'tile_t': 64, 'tile_n': 32, 'tile_k': 256}
  [attention] k1_attention(q<-add_13, k<-add_32, v<-mul_59) -> permute_7 attrs={'batch': 2, 'seq': 256, 'heads': 4, 'head_dim': 64, 'scale': 0.125} est=23,846cyc tiles={'tile_m': 32, 'tile_n': 128}
  [gemm] k2_gemm(lhs<-permute_7, w0<-permute_8, in0<-x) -> add_3 gemms=1 sumsq=False epilogue=1 est=12,249cyc tiles={'tile_t': 32, 'tile_n': 64, 'tile_k': 128}
  [gemm] k3_gemm(lhs<-add_3, w0<-permute_9_norm_65, w1<-permute_10_norm_69) -> mul_10 gemms=2 sumsq=True epilogue=13 est=34,991cyc tiles={'tile_t': 64, 'tile_n': 64, 'tile_k': 256}
  [gemm] k4_gemm(lhs<-mul_10, w0<-permute_11, in0<-add_3) -> add_5 gemms=1 sumsq=False epilogue=1 est=20,155cyc tiles={'tile_t': 32, 'tile_n': 64, 'tile_k': 128}
```