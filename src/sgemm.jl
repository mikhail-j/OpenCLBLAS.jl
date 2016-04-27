#=**
* High level implementation of single precision GEMM BLAS function.
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

module OpenCLBLAS

export sgemm, sgemm!

const libclblas = Libdl.find_library(["clBLAS.dll"],["C:\\AMD\\acml6.1.0.33\\ifort64\\lib\\"])
const libopencl = Libdl.find_library(["OpenCL64"],["C:\\Program Files\\NVIDIA Corporation\\OpenCL\\"])
if (isempty(libclblas))
	print("clBLAS can't be found!")
end

include("cl_typedef.jl")
include("clblas_typedef.jl")
include("cl_functions.jl")

function sgemm(tA::Char,tB::Char, alpha::cl_float, A::Array{cl_float,2}, B::Array{cl_float,2})
	local C = zeros(Float32, length(A[:,1]), length(B[1,:]))
	sgemm!(tA::Char,tB::Char, alpha::cl_float, A::Array{cl_float,2}, B::Array{cl_float,2},cl_float(0), C)
	return C
end

function sgemm(tA::Char,tB::Char, A::Array{cl_float,2}, B::Array{cl_float,2})
	local C = zeros(Float32, length(A[:,1]), length(B[1,:]))
	sgemm!(tA::Char,tB::Char, cl_float(1), A::Array{cl_float,2}, B::Array{cl_float,2},cl_float(0), C)
	return C
end

function sgemm!(tA::Char,tB::Char, alpha::cl_float, A::Array{cl_float,2}, B::Array{cl_float,2}, beta::cl_float, C::Array{cl_float,2})

	local props = vec(convert(Array{cl_context_properties, 2}, [CL_CONTEXT_PLATFORM 0 0]))
	devs = Array(cl_device_id, 1)
	devs[1] = clGetFirstGPU()
	local platform = clGetGPUPlatform(devs[1])
	props[2] = Base.cconvert(cl_context_properties,platform)
	err = Array(cl_int, 1)
	local ctx = clCreateContext(props,1,devs[1],C_NULL,C_NULL,err)
	statusCheck(err[1])
	err = Array(cl_int, 1)

	local queue = Array(cl_command_queue, 1)
	queue[1] = clCreateCommandQueue(ctx, devs[1], cl_command_queue_properties(0), err)
	statusCheck(err[1])
	################################	create arrays
	A1 = vec(A)
	B1 = vec(B)
	C1 = vec(C)
	M = Csize_t(length(A[:,1]))
	K = Csize_t(length(B[:,1]))
	N = Csize_t(length(B[1,:]))
	order = clblasColumnMajor		##julia uses column major
	local transA::clblasTranspose
	local transB::clblasTranspose
	if (tA == 'N')
		transA = clblasNoTrans;
	elseif (tA == 'T')
		transA = clblasTrans;
	end
	if (tB == 'N')
		transB = clblasNoTrans;
	elseif (tB == 'T')
		transB = clblasNoTrans;
	end
	statusCheck(clblasSetup())
	#statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * K * sizeof(cl_float), C_NULL, err)
	##println(M * K * sizeof(cl_float))
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])
	##println(K * N * sizeof(cl_float))
	err = Array(cl_int, 1)
	bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])

	event = Array(cl_event, 1)
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), M * K * sizeof(cl_float), A1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufB, CL_TRUE, Csize_t(0), K * N * sizeof(cl_float), B1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufC, CL_TRUE, Csize_t(0), M * N * sizeof(cl_float), C1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))

	event[1] = C_NULL
	statusCheck(clblasSgemm(clblasColumnMajor, clblasNoTrans, clblasNoTrans, M, N, K,
	                         alpha, bufA, 0, M,
	                         bufB, 0, K, beta,
	                         bufC, 0, M,
	                         1, queue, 0, C_NULL, event))
	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))

	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufC, CL_TRUE, Csize_t(0), length(C1)*sizeof(cl_float), C1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))

	statusCheck(clReleaseMemObject(bufC))
	statusCheck(clReleaseMemObject(bufB))
	statusCheck(clReleaseMemObject(bufA))
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
	Base.gc()
end

end