#=**
* Ported from OpenCLBLAS zHEMV example (example_zhemv.c)
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


function clblasZhemv(order,uplo,N,alpha,A,offA,lda,X,offX,incX,beta,Y,offY,incY,ncq,cq,ne,wle,e)
	return ccall((:clblasZhemv, libclblas), cl_int, (clblasOrder,
		clblasUplo,
		Csize_t,#N
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
		order,uplo,N,alpha,A,offA,lda,X,offX,incX,beta,Y,offY,incY,ncq,cq,ne,wle,e)
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
	A =	convert(Array{clblasDoubleComplex,2}, [
hcat(clblasDoubleComplex( 1.0, 00.0), clblasDoubleComplex( 2.0, 02.0), clblasDoubleComplex( 4.0,  4.0), clblasDoubleComplex( 7.0,  7.0), clblasDoubleComplex(11.0, 11.0));
hcat(clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex( 3.0, 03.0), clblasDoubleComplex( 5.0,  5.0), clblasDoubleComplex( 8.0,  8.0), clblasDoubleComplex(12.0, 12.0));
hcat(clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex( 6.0,  6.0), clblasDoubleComplex( 9.0,  9.0), clblasDoubleComplex(13.0, 13.0));
hcat(clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex(10.0, 10.0), clblasDoubleComplex(14.0, 14.0));
hcat(clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex(00.0, 00.0), clblasDoubleComplex(15.0, 15.0))])
	X = convert(Array{clblasDoubleComplex,1}, [clblasDoubleComplex(1.0, 0.0),clblasDoubleComplex(2.0, 0.0),clblasDoubleComplex(3.0, 0.0),clblasDoubleComplex(4.0, 0.0),clblasDoubleComplex(5.0, 0.0)]);
	Y = convert(Array{clblasDoubleComplex,1}, [clblasDoubleComplex(1.0, 0.0),clblasDoubleComplex(2.0, 0.0),clblasDoubleComplex(3.0, 0.0),clblasDoubleComplex(4.0, 0.0),clblasDoubleComplex(5.0, 0.0)]);

	N = Csize_t(size(A,1));
	lda = Csize_t(N);
	
	order = clblasColumnMajor;		##julia uses column major
	uplo = clblasUpper;
	incX = cl_int(1);
	incY = cl_int(1);
	alpha = zeros(clblasDoubleComplex, 1);
	alpha[1] = clblasDoubleComplex(10,10);
	beta = zeros(clblasDoubleComplex, 1);
	beta[1] = clblasDoubleComplex(20,20);
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * lda * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), N * lda * sizeof(clblasDoubleComplex), A.', cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufX, CL_TRUE, Csize_t(0), N * sizeof(clblasDoubleComplex), X, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory
	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufY, CL_TRUE, Csize_t(0), N * sizeof(clblasDoubleComplex), Y, cl_uint(0), C_NULL, event))
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

	statusCheck(clblasZhemv(order, uplo, N, alpha, bufA, 0, lda,
		bufX, 0, incX, beta,
		bufY, 0, incY, 1, queue, 0, C_NULL, event))

	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	Y2=Array(clblasDoubleComplex, length(Y))
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufY, CL_TRUE, Csize_t(0), N * sizeof(clblasDoubleComplex), Y2, cl_uint(0), C_NULL, event))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory

	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufY))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufX))
	statusCheck(clFlush(queue[1]))
	statusCheck(clReleaseMemObject(bufA))
	statusCheck(clFlush(queue[1]))
	#statusCheck(clGetMemObjectInfo(bufA, CL_MEM_REFERENCE_COUNT, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
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
	return Y2
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end