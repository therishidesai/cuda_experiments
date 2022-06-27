#!/usr/bin/env python3
import os
import time
import numpy as np

# numpy matrix multiplication
# write A, B, and C to a file so it can be used by CUDA program for validation
# Took this from @geohot: https://github.com/geohot/tinygrad/blob/gemm/extra/gemm/gemm.py

N = 4096
if __name__ == "__main__":
  # N^2
  A = np.random.randn(N, N).astype(np.float32)
  # N^2
  B = np.random.randn(N, N).astype(np.float32)

  # 2N compute in N^2 output cells
  flop = 2*N*N*N
  #print(f"{flop / 1e9:.2f} GFLOP")

  for i in range(1):
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s = et-st
    print(f"{flop/s * 1e-9:.2f} GFLOP/S, {s*1e3:.2f} ms")

  with open("/tmp/matmul", "wb") as f:
    f.write(A.data)
    f.write(B.data)
    f.write(C.astype(np.float32).data)
