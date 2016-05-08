#=**
* Ported from OpenCLBLAS sROTMG example (example_srotmg.c)
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


function clblasSrotmg(D1,offD1,D2,offD2,X1,offX1,Y1,offY1,param,offparam,ncq,cq,ne,wle,e)
	return ccall((:clblasSrotmg, libclblas), clblasStatus, (cl_mem,
		Csize_t,
		cl_mem,
		Csize_t,
		cl_mem,
		Csize_t,
		cl_mem,
		Csize_t,
		cl_mem,
		Csize_t,
		cl_uint,
		Ptr{cl_command_queue},
		cl_uint,
		Ptr{cl_event},
		Ptr{cl_event}),
		D1,offD1,D2,offD2,X1,offX1,Y1,offY1,param,offparam,ncq,cq,ne,wle,e)
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
	D1 = collect(cl_float(10));
	D2 = collect(cl_float(21));
	X1 = collect(cl_float(1));
	Y1 = collect(cl_float(-1));
	PARAM = convert(Array{cl_float, 1}, [-1,10,12,20,2]);

	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufD1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufD2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufX1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufY1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufP = clCreateBuffer(ctx, CL_MEM_READ_WRITE, length(PARAM) * sizeof(cl_float), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufD1, CL_TRUE, Csize_t(0), sizeof(cl_float), D1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufD2, CL_TRUE, Csize_t(0), sizeof(cl_float), D2, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufX1, CL_TRUE, Csize_t(0), sizeof(cl_float), X1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufY1, CL_TRUE, Csize_t(0), sizeof(cl_float), Y1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufP, CL_TRUE, Csize_t(0), length(PARAM) * sizeof(cl_float), PARAM, cl_uint(0), C_NULL, event))
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

	statusCheck(clblasSrotmg(bufD1, 0, bufD2, 0, bufX1, 0, bufY1, 0, bufP, 0, 1, queue, 0, C_NULL, event))

	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufD1, CL_TRUE, Csize_t(0), sizeof(cl_float), D1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufD2, CL_TRUE, Csize_t(0), sizeof(cl_float), D2, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufX1, CL_TRUE, Csize_t(0), sizeof(cl_float), X1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufY1, CL_TRUE, Csize_t(0), sizeof(cl_float), Y1, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufP, CL_TRUE, Csize_t(0), length(PARAM) * sizeof(cl_float), PARAM, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	println(string("D1: ", D1));
	println(string("D2: ", D2));
	println(string("X1: ", X1));
	println(string("Y1: ", Y1));
	println(string("PARAM: ", PARAM));

	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufP))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufY1))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufX1))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufD2))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufD1))
	statusCheck(clFlush(queue[1]))
	#statusCheck(clGetMemObjectInfo(bufA, CL_MEM_REFERENCE_COUNT, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	clblasTeardown()
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseCommandQueue(queue[1]))
	statusCheck(clReleaseContext(ctx))
	bufP = C_NULL
	bufY1 = C_NULL
	bufX1 = C_NULL
	bufD2 = C_NULL
	bufD1 = C_NULL
	queue[1] = C_NULL
	event[1] = C_NULL
	ctx = C_NULL
	devs[1] = C_NULL
	Base.gc()		##not sure if julia has been garbage collecting, now is a good time though
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end