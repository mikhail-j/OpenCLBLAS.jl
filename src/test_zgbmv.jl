#=**
* Ported from OpenCLBLAS zGBMV example (example_sgemm.c)
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

function clblasZgbmv(o,t,M,N,kl,ku,alpha,A,oA,lda,X,oX,iX,beta,Y,oY,iY,ncq,cq,ne,wle,e)
	return ccall((:clblasZgbmv, libclblas), clblasStatus, (clblasOrder,
		clblasTranspose,
		Csize_t,
		Csize_t,
		Csize_t,
		Csize_t,
		Ptr{clblasDoubleComplex},#alpha#treating this as a pointer fixed a segmentation fault
		cl_mem,#A
		Csize_t,
		Csize_t,
		cl_mem,#X
		Csize_t,
		cl_int,
		Ptr{clblasDoubleComplex},#beta#treating this as a pointer fixed a segmentation fault
		cl_mem,#Y
		Csize_t,
		cl_int,
		cl_uint,
		Ptr{cl_command_queue},
		cl_uint,
		Ptr{cl_event},
		Ptr{cl_event}),
		o,t,M,N,kl,ku,alpha,A,oA,lda,X,oX,iX,beta,Y,oY,iY,ncq,cq,ne,wle,e)
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
	local A = convert(Array{clblasDoubleComplex,2}, [hcat(00, 12, 13, 14);hcat(21, 22, 23, 24);hcat(31, 32, 33, 34);hcat(41, 42, 43, 00);hcat(51, 62, 00, 00)]);
	local X = convert(Array{clblasDoubleComplex,1}, [11,21,31,41,51])
	local Y = convert(Array{clblasDoubleComplex,1}, [11,21,31,41,51])
	local A1 = vec(A')
	local M = Csize_t(size(A)[1])
	#println(string("M: ", M));
	local N = M
	#println(string("N: ", N));
	local KL = Csize_t(1)
	local KU = Csize_t(2)
	local lda = Csize_t(KL + KU + 1)
	
	local order = clblasColumnMajor		##julia uses column major
	local alpha = Array(clblasDoubleComplex, 1)
	alpha[1] = convert(clblasDoubleComplex, 10)
	local beta = Array(clblasDoubleComplex, 1)
	beta[1] = convert(clblasDoubleComplex, 20)
	local trans = clblasNoTrans;
	local incX =  cl_int(1)
	local incY =  cl_int(1)
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * lda * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), M * lda * sizeof(clblasDoubleComplex), A1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufX, CL_TRUE, Csize_t(0), N * sizeof(clblasDoubleComplex), X, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufY, CL_TRUE, Csize_t(0), M * sizeof(clblasDoubleComplex), Y, cl_uint(0), C_NULL, event))
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
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufY, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufY memory object size: ", Int32(ref_count[1])))
	ref_count = 0
=====#
	event[1] = C_NULL
	statusCheck(clblasZgbmv(order, trans, M, N, KL, KU, alpha, bufA, 0, lda, bufX, 0, incX, beta, bufY, 0, incY, 1, queue, 0, C_NULL, event))
	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory


	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufY, CL_TRUE, Csize_t(0), M * sizeof(clblasDoubleComplex), Y, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufY))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufX))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufA))
	statusCheck(clFlush(queue[1]))

	clblasTeardown()
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseCommandQueue(queue[1]))
	statusCheck(clReleaseContext(ctx))
	bufY = C_NULL
	bufX = C_NULL
	bufA = C_NULL
	queue[1] = C_NULL
	event[1] = C_NULL
	ctx = C_NULL
	devs[1] = C_NULL
	Base.gc()		##not sure if julia has been garbage collecting, now is a good time though
	return Y
end

if (!isempty(libclblas) && !isempty(libopencl))
	#=
	A =	convert(Array{clblasDoubleComplex,2}, [hcat(00, 12, 13, 14);hcat(21, 22, 23, 24);hcat(31, 32, 33, 34);hcat(41, 42, 43, 00);hcat(51, 62, 00, 00)]);
	X = convert(Array{clblasDoubleComplex,1}, [11,21,31,41,51])
	Y = convert(Array{clblasDoubleComplex,1}, [11,21,31,41,51])
	Base.BLAS.gbmv!('N',5,1,2,Float32(10),A',X,Float32(20),Y)
	println(Y)
	=#
	main()
end