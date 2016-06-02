#=**
* Ported from OpenCLBLAS cHEMM example (example_chemm.c)
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


function clblasChemm(order,side,uplo,M,N,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
	return ccall((:clblasChemm, libclblas), cl_int, (clblasOrder,
		clblasSide,
		clblasUplo,
		Csize_t,#M
		Csize_t,#N
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
		order,side,uplo,M,N,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
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
	A =	convert(Array{clblasFloatComplex,2}, [
hcat(clblasFloatComplex(11, 12), clblasFloatComplex(-1, -1), clblasFloatComplex(-1, -1), clblasFloatComplex(-1, -1));
hcat(clblasFloatComplex(21, 22), clblasFloatComplex(22, 23), clblasFloatComplex(-1, -1), clblasFloatComplex(-1, -1));
hcat(clblasFloatComplex(31, 32), clblasFloatComplex(32, 33), clblasFloatComplex(33, 34), clblasFloatComplex(-1, -1));
hcat(clblasFloatComplex(41, 61), clblasFloatComplex(42, 62), clblasFloatComplex(43, 73), clblasFloatComplex(44, 23))]);
	B = convert(Array{clblasFloatComplex,2}, [
	hcat(clblasFloatComplex(11, -21),  clblasFloatComplex(-12, 23), clblasFloatComplex(13, 33));
    hcat(clblasFloatComplex(21, 12),   clblasFloatComplex(22, -10), clblasFloatComplex(23,  5));
    hcat(clblasFloatComplex(31, 1),    clblasFloatComplex(-32, 65), clblasFloatComplex(33, -1));
    hcat(clblasFloatComplex(1, 41),    clblasFloatComplex(-33, 42), clblasFloatComplex(12, 43))]);
	C = convert(Array{clblasFloatComplex,2}, [
	hcat(clblasFloatComplex(11, 11),  clblasFloatComplex(-12, 12), clblasFloatComplex(13, 33));
    hcat(clblasFloatComplex(21, -32), clblasFloatComplex(22,  -1), clblasFloatComplex(23,  0));
    hcat(clblasFloatComplex(31, 13),  clblasFloatComplex(-32, 78), clblasFloatComplex(33, 45));
    hcat(clblasFloatComplex(41, 14),  clblasFloatComplex(0,   42), clblasFloatComplex(43, -1))]);

	M = Csize_t(size(B,1));
	N = Csize_t(size(B,2));
	lda = Csize_t(M);
	ldb = Csize_t(N);
	ldc = Csize_t(N);
	
	order = clblasRowMajor;		##julia uses column major
	side = clblasLeft;
	uplo = clblasLower;
	incX = cl_int(1);
	incY = cl_int(1);
	alpha = clblasFloatComplex(10,10);
	beta = clblasFloatComplex(20,20);
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * M * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * N * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), M * M * sizeof(clblasFloatComplex), A.', cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufB, CL_TRUE, Csize_t(0), M * N * sizeof(clblasFloatComplex), B.', cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufC, CL_TRUE, Csize_t(0), M * N * sizeof(clblasFloatComplex), C.', cl_uint(0), C_NULL, event))
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

	statusCheck(clblasChemm(order, side, uplo, M, N, alpha, bufA,
		0, lda, bufB, 0, ldb, beta, bufC, 0, ldc, 1, queue,
		0, C_NULL, event))

	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	C2=Array(clblasFloatComplex, length(C))
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufC, CL_TRUE, Csize_t(0), M * N * sizeof(clblasFloatComplex), C2, cl_uint(0), C_NULL, event))
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
	return reshape(C2, Int(N), Int(M)).'
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end