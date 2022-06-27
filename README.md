# cuda_experiments

Some simple CUDA C programs to learn how the GPU works.

## Programs

### Basic Matrix Multiply

Compile: 

```
nvcc src/gemm.cu -o gemm
```

Run example:

```
python3 src/gemm.py && ./gemm
```

The python code will generate two random numpy arrays and multiply them then save the 3 arrays to a file.
The C code will then read that file and do the same matrix multiply on the gpu and compare results.
