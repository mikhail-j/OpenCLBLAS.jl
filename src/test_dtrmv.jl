#=**
* Ported from OpenCLBLAS dTRMV example (example_strmv.c)
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

function clblasDtrmv(o,ul,t,d,N,A,offA,lda,X,offX,incX,sb,ncq,cq,ne,wle,e)
	return ccall((:clblasDtrmv, libclblas), cl_int, (clblasOrder,
		clblasUplo,
		clblasTranspose,
		clblasDiag,
		Csize_t,#N
		cl_mem,#A
		Csize_t,
		Csize_t,
		cl_mem,#X
		Csize_t,
		Cint,
		cl_mem,#scratch buffer
		cl_uint,
		Ptr{cl_command_queue},
		cl_uint,
		Ptr{cl_event},
		Ptr{cl_event}),
		o,ul,t,d,N,A,offA,lda,X,offX,incX,sb,ncq,cq,ne,wle,e)
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
	A = convert(Array{cl_double,2}, [hcat(11, 12, 13, 14, 15);hcat(0, 22, 23, 24, 25);hcat(0, 0, 33, 34, 35);hcat(0, 0, 0, 44, 45);hcat(0, 0, 0, 0, 55)]);
	X = convert(Array{cl_double,1}, [11,21,31,41,51]);
	N = Csize_t(size(X)[1]);
	lda = Csize_t(N);
	incX = cl_int(1);
	order = clblasColumnMajor;
	uplo = clblasUpper;
	trans = clblasTrans;
	diag = clblasUnit;
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())

	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * lda * sizeof(cl_double), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufX = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * sizeof(cl_double), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	sbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(cl_double), C_NULL, err)
	statusCheck(err[1])
	
	event = Array(cl_event,1);

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), N * lda * sizeof(cl_double), A, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufX, CL_TRUE, Csize_t(0), N * sizeof(cl_double), X, cl_uint(0), C_NULL, event))
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
	statusCheck(clGetMemObjectInfo(sbuf, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("sbuf memory object size: ", Int32(ref_count[1])))
	ref_count = 0
=====#
#=
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufA, CL_MEM_REFERENCE_COUNT, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufA reference count: ", Int32(ref_count[1])))
	ref_count = 0
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufX, CL_MEM_REFERENCE_COUNT, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufX reference count: ", Int32(ref_count[1])))
	ref_count = 0
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(sbuf, CL_MEM_REFERENCE_COUNT, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("sbuf reference count: ", Int32(ref_count[1])))
=#
	event[1] = C_NULL
	statusCheck(clblasDtrmv(order,uplo,trans,diag,N,bufA,Csize_t(0), lda, bufX, Csize_t(0), incX, sbuf, cl_int(1), queue, cl_uint(0), C_NULL, event))
	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory


	X1 = zeros(cl_double,Int(N))
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufX, CL_TRUE, Csize_t(0), N * sizeof(cl_double), X1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(sbuf))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufX))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufA))
	statusCheck(clFlush(queue[1]))

	clblasTeardown()
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseCommandQueue(queue[1]))
	statusCheck(clReleaseContext(ctx))
	sbuf = C_NULL
	bufA = C_NULL
	bufX = C_NULL
	trans = 0
	uplo = 0
	diag = 0
	order = 0
	queue[1] = C_NULL
	event[1] = C_NULL
	ctx = C_NULL
	devs[1] = C_NULL
	Base.gc()		##not sure if julia has been garbage collecting, now is a good time though
	return X1
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end