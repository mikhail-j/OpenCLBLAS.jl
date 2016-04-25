# OpenCLBLAS.jl

######OpenCL BLAS library wrapper for Julia with samples

#####This project is based off the work of these projects:

[OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl) by Jake Bolewski and Valentin Churavy  
[CUBLAS.jl](https://github.com/JuliaGPU/CUBLAS.jl) by Nick Henderson

##What is this project?

This project focuses on running OpenCL BLAS on GPU devices specifically, all OpenCL type definitions and functions were hand typed from cl.h and clBLAS.h header.  
  
Currently, I've only rewritten the example C program (single precision GEMM BLAS) provided with libclBLAS into test_sgemm.jl. 
I also made a separate high level function to manages all the memory involved with calling the C function clblasSgemm().  

As a result, the OpenCLBLAS.sgemm() function cleans up the buffers, events, contexts, and queues in the GPU memory.

##Progress

I'll add more OpenCLBLAS functions as I test them myself.

The following functions have been tested on a Windows x64 PC with an NVIDIA GPU:  
-**sGEMM**

##License

The license can be found in the file 'LICENSE'.