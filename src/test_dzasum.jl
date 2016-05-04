#=**
* Ported from OpenCLBLAS dzASUM example (example_sasum.c)
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

function clblasDzasum(N,as,offA,X,offX,incX,sb,ncq,cq,ne,wle,e)
	return ccall((:clblasDzasum, libclblas), cl_int, (Csize_t,#N
		cl_mem,#absolute sum
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
		N,as,offA,X,offX,incX,sb,ncq,cq,ne,wle,e)
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
	X = vec([clblasDoubleComplex(1),clblasDoubleComplex(2),clblasDoubleComplex(-11),clblasDoubleComplex(17),clblasDoubleComplex(5),clblasDoubleComplex(6),clblasDoubleComplex(81)]);
	N = Csize_t(length(X));
	incX = cl_int(1);
	local asum = zeros(cl_double, 1);
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	local bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	local sbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	local bufASUM = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufX, CL_TRUE, Csize_t(0), N * sizeof(clblasDoubleComplex), X, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory


#=================Check respective buffer sizes in GPU
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufX, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufX memory object size: ", Int32(ref_count[1])))
	ref_count = 0
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufASUM, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufASUM memory object size: ", Int32(ref_count[1])))
	ref_count = 0
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(sbuf, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("sbuf memory object size: ", Int32(ref_count[1])))
	ref_count = 0
=====#
	event[1] = C_NULL
	statusCheck(clblasDzasum(N, bufASUM, 0, bufX, 0, incX, sbuf, 1, queue, 0, C_NULL, event))
	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory



	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufASUM, CL_TRUE, Csize_t(0), sizeof(cl_double), asum, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

#=
	scratch = zeros(clblasDoubleComplex, N)
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], sbuf, CL_TRUE, Csize_t(0), N * sizeof(clblasDoubleComplex), scratch, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	println(scratch)
=#

	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(sbuf))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufASUM))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufX))
	statusCheck(clFlush(queue[1]))

	clblasTeardown()
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseCommandQueue(queue[1]))
	statusCheck(clReleaseContext(ctx))
	sbuf = C_NULL
	bufASUM = C_NULL
	bufX = C_NULL
	queue[1] = C_NULL
	event[1] = C_NULL
	ctx = C_NULL
	devs[1] = C_NULL
	Base.gc()		##not sure if julia has been garbage collecting, now is a good time though
	return asum[1]
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end