#=**
* Ported from OpenCLBLAS sTRSM example (example_strsm.c)
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


function clblasStrsm(o,side,ul,tA,diag,M,N,alpha,A,offA,lda,B,offB,ldb,ncq,cq,ne,wle,e)
	return ccall((:clblasStrsm, libclblas), cl_int, (clblasOrder,
		clblasSide,
		clblasUplo,
		clblasTranspose,
		clblasDiag,
		Csize_t,
		Csize_t,
		cl_float,#alpha
		cl_mem,#A
		Csize_t,
		Csize_t,
		cl_mem,#B
		Csize_t,
		Csize_t,
		cl_uint,
		Ptr{cl_command_queue},
		cl_uint,
		Ptr{cl_event},
		Ptr{cl_event}),
		o,side,ul,tA,diag,M,N,alpha,A,offA,lda,B,offB,ldb,ncq,cq,ne,wle,e)
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
	A =	convert(Array{cl_float,2}, [hcat(11, 12, 13, 14);hcat(0, 22, 23, 24);hcat(0,  0, 33, 34);hcat(0,  0,  0, 44)])
	B = convert(Array{cl_float,2}, [hcat(11, 12, 13, 14, 15);hcat(21, 22, 23, 24, 25);hcat(31, 32, 33, 34, 35);hcat(41, 42, 43, 44, 45)])

	M = Csize_t(size(B,1))
	N = Csize_t(size(B,2))
	lda = Csize_t(M)
	ldb = Csize_t(N)
	
	order = clblasRowMajor		##julia uses column major
	side = clblasLeft
	#alpha = Array(cl_float, 1)
	#alpha[1] = convert(cl_float, 10)
	alpha = convert(cl_float, 10)
	#println(string("alpha: ",alpha))
	transA = clblasNoTrans;
	uploA = clblasUpper;
	diagA = clblasNonUnit;
	off =  convert(Csize_t, 0)
	offA = convert(Csize_t, 0)
	offB = convert(Csize_t, 0)
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * M * sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), M * M * sizeof(cl_float), transposeNC(A), cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufB, CL_TRUE, Csize_t(0), M * N * sizeof(cl_float), transposeNC(B), cl_uint(0), C_NULL, event))
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

	statusCheck(clblasStrsm(order, side, uploA, transA, diagA, M, N,
							alpha, bufA, offA, lda, bufB, offB, ldb, 1, queue, 0,
							C_NULL, event))

	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	B2=Array(cl_float,length(B))
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufB, CL_TRUE, Csize_t(0), M * N * sizeof(cl_float), B2, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

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
	return transposeNC(reshape(B2, Int(N), Int(M)))
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end