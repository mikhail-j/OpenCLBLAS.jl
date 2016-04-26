# OpenCLBLAS.jl

######OpenCL BLAS library wrapper for Julia with samples


##What is this project?

This project focuses on running OpenCL BLAS with Julia matrices on GPU devices seamlessly, all OpenCL type definitions and functions were hand-typed from cl.h and clBLAS.h header.  

Currently, I've only rewritten the example C program (single precision GEMM BLAS) provided with libclBLAS into test_sgemm.jl. 
I also made a separate high level function to manages all the memory involved with calling the C function clblasSgemm().  

As a result, the OpenCLBLAS.sgemm() function cleans up the buffers, events, contexts, and queues in the GPU memory.

##Usage

The following example modifies the C variable (matrix).  

```Julia
include("cl_typedef.jl")
include("sgemm.jl")

A = convert(Array{cl_float,2}, [[11, 12, 13, 14, 15]';[21, 22, 23, 24, 25]';[31, 32, 33, 34, 35]';[41, 42, 43, 44, 45]'])
B = convert(Array{cl_float,2}, [[11, 12, 13]';[21, 22, 23]';[31, 32, 33]';[41, 42, 43]';[51, 52, 53]'])
C = convert(Array{cl_float,2}, [[11, 12, 13]';[21, 22, 23]';[31, 32, 33]';[41, 42, 43]'])
OpenCLBLAS.sgemm!('N','N',cl_float(10),A,B, cl_float(20), C)
```

##Progress

I'll add more OpenCLBLAS functions as I test them myself.

The following functions have been tested on a Windows x64 PC with an NVIDIA GPU:  
-**sGEMM**

#####This project is based off the work of these projects:

[OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl) by Jake Bolewski and Valentin Churavy  
[CUBLAS.jl](https://github.com/JuliaGPU/CUBLAS.jl) by Nick Henderson  

##License

The license can be found in the file 'LICENSE'.