# OpenCLBLAS.jl
OpenCL BLAS library wrapper for Julia with samples
============================

#####This project is based off the work of these projects:

[OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl)  
[CUBLAS.jl](https://github.com/JuliaGPU/CUBLAS.jl)

##What is this project?
----------------------------

This project focuses on running OpenCL BLAS on GPU devices specifically.
Currently, I've only rewrote the example C program (single precision GEMM BLAS) provided with libclBLAS into test_sgemm.jl.
As a result, I also made a separate high level function to manages all the memory involved with calling clblasSgemm(), this include removing all traces of the buffers in the GPU memory.

##Progress
----------------------------

I'll add more OpenCLBLAS functions as I test them myself.

The following functions have been tested on a Windows x64 PC with an NVIDIA GPU:
-**sGEMM**

##License
----------------------------

The license can be found in the file 'LICENSE'.