#=**
* Ported from OpenCLBLAS zHER2 example (example_zher2.c)
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


function clblasZher2(order,uplo,N,alpha,X,offx,incX,Y,offy,incy,A,offA,lda,ncq,cq,ne,wle,e)
	return ccall((:clblasZher2, libclblas), cl_int, (clblasOrder,
		clblasUplo,
		Csize_t,
		clblasDoubleComplex,
		cl_mem,#X
		Csize_t,
		cl_int,
		cl_mem,#Y
		Csize_t,
		cl_int,
		cl_mem,#A
		Csize_t,
		Csize_t,
		cl_uint,
		Ref{cl_command_queue},
		cl_uint,
		Ref{cl_event},
		Ref{cl_event}),
		order,uplo,N,alpha,X,offx,incX,Y,offy,incy,A,offA,lda,ncq,cq,ne,wle,e)
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
	#Hermitian matrix is a square matrix by definition
	A = [hcat(Complex{Float64}(11.0, 00.0), Complex{Float64}(12.0, 02.0), Complex{Float64}(13.0, 05.0), Complex{Float64}(14.0, 12.0), Complex{Float64}(15.0, 06.0));hcat(Complex{Float64}(00.0, 00.0), Complex{Float64}(22.0, 00.0), Complex{Float64}(23.0, 25.0), Complex{Float64}(24.0, 23.0), Complex{Float64}(25.0, 61.0));hcat(Complex{Float64}(00.0, 00.0), Complex{Float64}(00.0, 00.0), Complex{Float64}(33.0, 00.0), Complex{Float64}(34.0, 23.0), Complex{Float64}(35.0, 21.0));hcat(Complex{Float64}(00.0, 00.0), Complex{Float64}(00.0, 00.0), Complex{Float64}(00.0, 00.0), Complex{Float64}(44.0, 00.0), Complex{Float64}(45.0, 23.0));hcat(Complex{Float64}(00.0, 00.0), Complex{Float64}(00.0, 00.0), Complex{Float64}(00.0, 00.0), Complex{Float64}(00.0, 00.0), Complex{Float64}(55.0, 00.0))]
	X = [Complex{Float64}(11.0, 03.0),Complex{Float64}(01.0, 15.0),Complex{Float64}(30.0, 20.0),Complex{Float64}(01.0, 02.0),Complex{Float64}(11.0, 10.0)]
	Y = [Complex{Float64}(11.0, 03.0),Complex{Float64}(03.0, 05.0),Complex{Float64}(09.0, 00.0),Complex{Float64}(01.0, 02.0),Complex{Float64}(11.0, 00.0)]

	A1 = vec(A)

	#println(length(A1))

	N=Csize_t(length(X))
	order = clblasColumnMajor;
	uplo = clblasUpper;
	#uplo = clblasUpper;
	alpha = clblasDoubleComplex(10,2)
	#println(string("alpha: ",alpha))
	incX = convert(Csize_t, 1)
	incY = convert(Csize_t, 1)
	#lda = convert(Csize_t, N)
	#Now initialize OpenCLBLAS and buffers
	statusCheck(clblasSetup())
	statusCheck(clFlush(queue[1]))
	err = Array(cl_int, 1)
	bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * N * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	err = Array(cl_int, 1)
	bufY = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N * sizeof(clblasDoubleComplex), C_NULL, err)
	statusCheck(err[1])
	statusCheck(clFlush(queue[1]))
	
	event = Array(cl_event, 1)

	event[1] = C_NULL
	statusCheck(clEnqueueWriteBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), N * N * sizeof(clblasDoubleComplex), A1, cl_uint(0), C_NULL, event))
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
	ref_count = Array(Csize_t, 1)
	statusCheck(clGetMemObjectInfo(bufY, CL_MEM_SIZE, Csize_t(sizeof(ref_count)), ref_count, C_NULL))
	println(string("bufY memory object size: ", Int32(ref_count[1])))
	ref_count = 0
=====#
	event[1] = C_NULL

	statusCheck(clblasZher2(order, uplo, N, alpha, bufX, 0, incX, bufY, 0, incY, bufA, 0, N, 1, queue, 0, C_NULL, event))
	statusCheck(clFlush(queue[1]))
	statusCheck(clWaitForEvents(1,event))
	statusCheck(clReleaseEvent(event[1]))		#free the memory


	A2=vec(Array(clblasDoubleComplex,N*N))
	event[1] = C_NULL
	statusCheck(clEnqueueReadBuffer(queue[1], bufA, CL_TRUE, Csize_t(0), N*N*sizeof(clblasDoubleComplex), A2, cl_uint(0), C_NULL, event))
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
	queue[1] = C_NULL
	event[1] = C_NULL
	ctx = C_NULL
	devs[1] = C_NULL
	Base.gc()		##not sure if julia has been garbage collecting, now is a good time though
	return reshape(A2,Int(N),Int(N))
end

if (!isempty(libclblas) && !isempty(libopencl))
	main()
end