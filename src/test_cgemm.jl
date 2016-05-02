#=**
* Ported from OpenCLBLAS cGEMM example (example_sgemm.c)
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
#ccall((:function, “library”), return_type, (argtype,),arg)


function clblasCgemm(o,tA,tB,M,N,K,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
	return ccall((:clblasCgemm, libclblas), cl_int, (clblasOrder,
		clblasTranspose,
		clblasTranspose,
		Csize_t,
		Csize_t,
		Csize_t,
		clblasFloatComplex,
		cl_mem,
		Csize_t,
		Csize_t,
		cl_mem,
		Csize_t,
		Csize_t,
		clblasFloatComplex,
		#Base.cconvert(Ptr{Void}, Ref{cl_mem}),
		#Ref{cl_mem},
		cl_mem,
		Csize_t,
		Csize_t,
		cl_uint,
		Ref{cl_command_queue},
		cl_uint,
		#Ref{cl_event},
		#AMD's OpenCL driver (Windows 7 x64) throws invalid event if argument type is Ref{cl_event}
		Ptr{cl_event},
		Ptr{cl_event}),
		#Ptr{cl_event_info},
		#Ptr{cl_event_info}),
		o,tA,tB,M,N,K,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
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
	A =	convert(Array{cl_float2,2}, convert(Array{clblasFloatComplex,2}, [[11, 12, 13, 14, 15]';[21, 22, 23, 24, 25]';[31, 32, 33, 34, 35]';[41, 42, 43, 44, 45]']))
	B = convert(Array{cl_float2,2}, convert(Array{clblasFloatComplex,2}, [[11, 12, 13]';[21, 22, 23]';[31, 32, 33]';[41, 42, 43]';[51, 52, 53]']))
	C = convert(Array{cl_float2,2}, convert(Array{clblasFloatComplex,2}, [[11, 12, 13]';[21, 22, 23]';[31, 32, 33]';[41, 42, 43]']))
	A1 = vec(A)
	B1 = vec(B)
	C1 = vec(C)
	M = Csize_t(length(A[:,1]))
	K = Csize_t(length(B[:,1]))
	N = Csize_t(length(B[1,:]))
	
	order = clblasColumnMajor		##julia uses column major
	alpha = convert(clblasFloatComplex, 10)
	#println(string("alpha: ",alpha))
	beta = convert(clblasFloatComplex, 20)
	#println(string("beta: ",beta))
	transA = clblasNoTrans;
	transB = clblasNoTrans;
	off =  convert(Csize_t, 0)
	offA = convert(Csize_t, 0)
	offB = convert(Csize_t, 0)
	offC = convert(Csize_t, 0)
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * K * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(clblasFloatComplex), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), M * K * sizeof(clblasFloatComplex), A1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufB, CL_TRUE, Csize_t(0), K * N * sizeof(clblasFloatComplex), B1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufC, CL_TRUE, Csize_t(0), M * N * sizeof(clblasFloatComplex), C1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory


#=================Check respective buffer sizes in GPU
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufA, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufA memory object size: ", Int32(ref_count[1])))
	ref_count = 0
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufB, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufB memory object size: ", Int32(ref_count[1])))
	ref_count = 0
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufC, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufC memory object size: ", Int32(ref_count[1])))
	ref_count = 0
=====#
	event[1] = C_NULL
	#=
	statusCheck(clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, M, N, K,
	                         alpha, bufA, 0, K,
	                         bufB, 0, N, beta,
	                         bufC, 0, N,
	                         1, queue, 0, C_NULL, event))
=#
	statusCheck(clblasCgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans, M, N, K,
	                         alpha, bufA, 0, M,
	                         bufB, 0, K, beta,
	                         bufC, 0, M,
	                         1, queue, 0, C_NULL, event))
	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory


	C2=Array(clblasFloatComplex,length(C1))
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufC, CL_TRUE, Csize_t(0), length(C1)*sizeof(clblasFloatComplex), C2, cl_uint(0), C_NULL, event))
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
	#bufA = C_NULL
	#bufB = C_NULL
	#bufC = C_NULL
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
	return reshape(C2, Int(M), Int(N))
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end