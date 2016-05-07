#=**
* Ported from OpenCLBLAS cSYR2K example (example_ssyr2k.c)
* Qijia (Michael) Jin
* @version 0.0.1
*
* Copyright (C) 2016  Qijia (Michael) Jin
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*=#

const libclblas = Libdl.find_library(["clBLAS","libclBLAS"],["C:\\AMD\\clBLA-2.10.0\\bin","C:\\AMD\\acml6.1.0.33\\ifort64\\lib\\"])
#const libopencl = Libdl.find_library(["libOpenCL","OpenCL"],["."])
const libopencl = Libdl.find_library(["OpenCL64","OpenCL"],["C:\\Program Files\\NVIDIA Corporation\\OpenCL\\","C:\\Program Files (x86)\\AMD APP SDK\\2.9-1\\bin\\x86_64"])
if (isempty(libclblas))
	print("clBLAS can't be found!")
end
include("cl_typedef.jl")
include("clblas_typedef.jl")
include("cl_functions.jl")
include("clblas_functions.jl")
include("custom_transpose.jl")
#ccall((:function, “library”), return_type, (argtype,),arg)


function clblasCsyr2k(o,ul,t,N,K,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
	return ccall((:clblasCsyr2k, libclblas), cl_int, (clblasOrder,
		clblasUplo,
		clblasTranspose,
		Csize_t,#N
		Csize_t,#K
		clblasFloatComplex,#alpha
		cl_mem,#A
		Csize_t,
		Csize_t,
		cl_mem,#B
		Csize_t,
		Csize_t,
		clblasFloatComplex,#beta
		cl_mem,#C
		Csize_t,
		Csize_t,
		cl_uint,
		Ptr{cl_command_queue},
		cl_uint,
		Ptr{cl_event},
		Ptr{cl_event}),
		o,ul,t,N,K,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
end

function main()

	local props = vec(convert(Array{cl_context_properties, 2}, [CL_CONTEXT_PLATFORM 0 0]))
	devs = Array(cl_device_id, 1)
	devs[1] = clGetFirstGPU()
	local platform = clGetGPUPlatform(devs[1])

	println(string("Selected GPU: ",clGetDeviceVendor(devs[1])), " ", clGetDeviceName(devs[1]))
	props[2] = Base.cconvert(cl_context_properties,platform)
	err = Array(cl_int, 1)
	local ctx = clCreateContext(props,1,devs[1],C_NULL,C_NULL,err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	local queue = Array(cl_command_queue, 1)
	queue[1] = clCreateCommandQueue(ctx, devs[1], cl_command_queue_properties(0), err)
	statusCheck(err[1])
	################################	create arrays
	A =	convert(Array{clblasFloatComplex,2}, [hcat(11, 12, 13, 14);hcat(21, 22, 23, 24);hcat(31, 32, 33, 34);hcat(41, 42, 43, 44);hcat(51, 52, 53, 54)]);
	B =	convert(Array{clblasFloatComplex,2}, [hcat(11, 12, 13, 14);hcat(21, 22, 23, 24);hcat(31, 32, 33, 34);hcat(41, 42, 43, 44);hcat(51, 52, 53, 54)]);
	C = convert(Array{clblasFloatComplex,2}, [hcat(11, 12, 13, 14, 15);hcat(12, 22, 23, 24, 25);hcat(13, 23, 33, 34, 35);hcat(14, 24, 34, 44, 45);hcat(15, 25, 35, 45, 55)]);

	N = Csize_t(size(C,1));
	K = Csize_t(size(A,2));
	#println(N)
	lda = Csize_t(K);
	ldb = Csize_t(K);
	ldc = Csize_t(N);
	alpha = clblasFloatComplex(10);
	beta = clblasFloatComplex(20);

	order = clblasRowMajor		##julia uses column major
	transAB = clblasNoTrans;
	uplo = clblasUpper;
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * N * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), N * K * sizeof(clblasFloatComplex), transposeNC(A), cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufB, CL_TRUE, Csize_t(0), N * K * sizeof(clblasFloatComplex), transposeNC(B), cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufC, CL_TRUE, Csize_t(0), N * N * sizeof(clblasFloatComplex), transposeNC(C), cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory


#=================Check respective buffer sizes in GPU
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufA, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufA memory object size: ", Int32(ref_count[1])))
	ref_count = 0
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufX, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufX memory object size: ", Int32(ref_count[1])))
	ref_count = 0
=====#
	event[1] = C_NULL

	statusCheck(clblasCsyr2k(order, uplo, transAB, N, K, alpha, bufA, 0, lda, bufB, 0, ldb, beta, bufC, 0, ldc, 1, queue, 0, C_NULL, event))

	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	C2=Array(clblasFloatComplex,length(C))
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufC, CL_TRUE, Csize_t(0), N * N * sizeof(clblasFloatComplex), C2, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufC))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufB))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufA))
	statusCheck(clFlush(queue[1]))
	#statusCheck(clGetMemObjectInfo(bufA, CL_MEM_REFERENCE_COUNT, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	clblasTeardown()
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseCommandQueue(queue[1]))
	statusCheck(clReleaseContext(ctx))
	bufC = C_NULL
	bufB = C_NULL
	bufA = C_NULL
	queue[1] = C_NULL
	event[1] = C_NULL
	ctx = C_NULL
	devs[1] = C_NULL
	Base.gc()		##not sure if julia has been garbage collecting, now is a good time though
	#.' should do the same as transposeNC()
	return reshape(C2, Int(N), Int(N)).'
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end