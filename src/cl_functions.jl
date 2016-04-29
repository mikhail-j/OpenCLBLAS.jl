#=**
* OpenCL and OpenCL functions from "https://www.khronos.org/registry/cl/sdk/1.0/docs/"
* Includes several higher level functions like clGetDeviceName() or clGetDeviceVendor()
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

include("cl_typedef.jl")
include("clblas_typedef.jl")

function statusParse(clstatus)
	if clstatus == 0#          
		return "CL_SUCCESS"
	elseif clstatus == -1#    
		return "CL_DEVICE_NOT_FOUND"
	elseif clstatus == -30#    
		return "CL_INVALID_VALUE"
	elseif clstatus == -36#    
		return "CL_INVALID_COMMAND_QUEUE"
	elseif clstatus ==-34#     
		return "CL_INVALID_CONTEXT"
	elseif clstatus == -38#    
		return "CL_INVALID_MEM_OBJECT"
	elseif clstatus == -33#    
		return "CL_INVALID_DEVICE"
	elseif clstatus == -57#    
		return "CL_INVALID_EVENT_WAIT_LIST"
	elseif clstatus == -5#     
		return "CL_OUT_OF_RESOURCES"
	elseif clstatus == -6#     
		return "CL_OUT_OF_HOST_MEMORY"
	elseif clstatus == -59#    
		return "CL_INVALID_OPERATION"
	elseif clstatus == -3#     
		return "CL_COMPILER_NOT_AVAILABLE"
	elseif clstatus == -11#    
		return "CL_BUILD_PROGRAM_FAILURE"
	elseif clstatus == -1024#  
		return "OpenCL BLAS is not implemented!"
	elseif clstatus == -1023#  
		return "OpenCL BLAS is not initialized!"
	elseif clstatus == -1022#  
		return "Matrix A is not a valid memory object!"
	elseif clstatus == -1021#  
		return "Matrix B is not a valid memory object!"
	elseif clstatus == -1020#  
		return "Matrix C is not a valid memory object!"
	elseif clstatus == -1019#  
		return "Vector X is not a valid memory object!"
	elseif clstatus == -1018#  
		return "Vector Y is not a valid memory object!"
	elseif clstatus == -1017#  
		return "An input dimension (M,N,K) is invalid!"
	elseif clstatus == -1016#  
		return "Leading dimension A must not be less than the size of the first dimension!"
	elseif clstatus == -1015#  
		return "Leading dimension B must not be less than the size of the second dimension!"
	elseif clstatus == -1014#  
		return "Leading dimension C must not be less than the size of the third dimension!"
	elseif clstatus == -1013#  
		return "The increment for a vector X must not be 0!"
	elseif clstatus == -1012#  
		return "The increment for a vector Y must not be 0!"
	elseif clstatus == -1011#  
		return "The memory object for Matrix A is too small!"
	elseif clstatus == -1010#  
		return "The memory object for Matrix B is too small!"
	elseif clstatus == -1009#  
		return "The memory object for Matrix C is too small!"
	elseif clstatus == -1008#  
		return "The memory object for Vector X is too small!"
	elseif clstatus == -1007#  
		return "The memory object for Vector Y is too small!"
	else
		return string("Error code: ", clstatus)
	end
end
function statusCheck(clstatus)
	if clstatus == clblasSuccess
		#println("SUCCESS")
		return
	else
		warn("CLBLAS Error:")
		Base.show_backtrace(Base.STDOUT,backtrace())
		println()
		throw(statusParse(clstatus))
	end
end

function clblasSetup()
	return ccall((:clblasSetup, libclblas), clblasStatus, ())
end
function clblasTeardown()
	ccall((:clblasTeardown, libclblas), Void, ())
end

function clblasSgemm(o,tA,tB,M,N,K,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
	return ccall((:clblasSgemm, libclblas), cl_int, (clblasOrder,
		clblasTranspose,
		clblasTranspose,
		Csize_t,
		Csize_t,
		Csize_t,
		cl_float,
		cl_mem,
		Csize_t,
		Csize_t,
		cl_mem,
		Csize_t,
		Csize_t,
		cl_float,
		#Base.cconvert(Ptr{Void}, Ref{cl_mem}),
		#Ref{cl_mem},
		cl_mem,
		Csize_t,
		Csize_t,
		cl_uint,
		Ref{cl_command_queue},
		cl_uint,
		Ref{cl_event},
		Ref{cl_event}),
		#Ptr{cl_event_info},
		#Ptr{cl_event_info}),
		o,tA,tB,M,N,K,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
end
function clGetPlatformIDs(entries, p, np)
	#return ccall((:clGetPlatformIDs, libopencl), cl_int, (cl_uint,Ptr{cl_platform_id},Ptr{cl_uint}),entries,p,np)
	return ccall((:clGetPlatformIDs, libopencl), cl_int, (cl_uint,Ptr{cl_platform_id},Base.cconvert(Ptr{Void},Ref{cl_uint})),entries,p,np)
end
function clGetDeviceIDs(p, dtype, entries, d, nd)
	#return ccall((:clGetPlatformIDs, libopencl), cl_int, (cl_uint,Ptr{cl_platform_id},Ptr{cl_uint}),entries,p,np)
	return ccall((:clGetDeviceIDs, libopencl), cl_int, (cl_platform_id, cl_device_type, cl_uint,Ptr{cl_device_id},Base.cconvert(Ptr{Void},Ref{cl_uint})),
		p,dtype,entries,d,nd)
end
function clGetDeviceInfo(d,pn,pvs,pv,pvsr)
	return ccall((:clGetDeviceInfo, libopencl), cl_int,(cl_device_id,cl_device_info,Csize_t,Ptr{Void},Base.cconvert(Ptr{Void},Ref{Csize_t})),
		d,pn,pvs,pv,pvsr)
end
function clCreateContext(properties,num,dev,notify,ud, err)
	return ccall((:clCreateContext, libopencl), cl_context, (Base.cconvert(Ptr{Void}, Ref{cl_context_properties}),
		cl_uint,
		Base.cconvert(Ptr{Void}, Ref{cl_device_id}),
		Ptr{Void},
		Ptr{Void},
		Base.cconvert(Ptr{Void},Ref{cl_int})),
		properties,num,dev,notify,ud,err)
end
function clReleaseContext(ctx)
	return ccall((:clReleaseContext, libopencl), cl_int, (cl_context,),ctx)
end
function clCreateCommandQueue(ctx, dev, properties, err)
	return ccall((:clCreateCommandQueue, libopencl), cl_command_queue, (cl_context,
		cl_device_id,
		cl_context_properties,
		Base.cconvert(Ptr{Void},Ref{cl_int})),
		ctx, dev, properties, err)
end
function clReleaseCommandQueue(cq)
	return ccall((:clReleaseCommandQueue, libopencl), cl_int, (cl_command_queue,), cq)
end
function clCreateBuffer(ctx, flags, length, host_ptr, err)
	return ccall((:clCreateBuffer, libopencl), cl_mem, (cl_context,
		cl_mem_flags,
		Csize_t,
		Ptr{Void},
		Base.cconvert(Ptr{Void}, Ref{cl_int})),
		ctx, flags, length, host_ptr, err)
end
function clReleaseMemObject(b)
	return ccall((:clReleaseMemObject, libopencl), cl_int, (cl_mem,), b)
end
function clEnqueueWriteBuffer(q, b, isBlocking, off, cb, host_ptr, ne, wle, e)
	return ccall((:clEnqueueWriteBuffer, libopencl), cl_int, (cl_command_queue,
		cl_mem,
		cl_bool,
		Csize_t,
		Csize_t,
		Ptr{Void},
		cl_uint,
		Base.cconvert(Ptr{Void}, Ref{cl_event}),
		Base.cconvert(Ptr{Void}, Ref{cl_event})),
		q, b, isBlocking, off, cb, host_ptr, ne, wle, e)
end
function clEnqueueReadBuffer(q, b, isBlocking, off, cb, host_ptr, ne, wle, e)
	return ccall((:clEnqueueReadBuffer, libopencl), cl_int, (cl_command_queue,
		cl_mem,
		cl_bool,
		Csize_t,
		Csize_t,
		Ptr{Void},
		cl_uint,
		Base.cconvert(Ptr{Void}, Ref{cl_event}),
		Base.cconvert(Ptr{Void}, Ref{cl_event})),
		q, b, isBlocking, off, cb, host_ptr, ne, wle, e)
end
function clWaitForEvents(ne, el)
	return ccall((:clWaitForEvents, libopencl), cl_int,(cl_uint, Base.cconvert(Ptr{Void},Ref{cl_event})), ne, el)
end
function clReleaseEvent(e)
	return ccall((:clReleaseEvent, libopencl), cl_int, (cl_event,), e)
end
function clGetMemObjectInfo(mo, name, input_size, param, return_size)
	return ccall((:clGetMemObjectInfo, libopencl), cl_int, (cl_mem,cl_mem_info,Csize_t, Ptr{Void}, Csize_t),
		mo, name, input_size, param, return_size)
end
function clFlush(cq)
	return ccall((:clFlush, libopencl), cl_int,(cl_command_queue,), cq)
end
function clFinish(cq)
	return ccall((:clFinish, libopencl), cl_int,(cl_command_queue,), cq)
end
function flatten(matrix)
	return vec(matrix')
end
function unflatten(matrix, r, c)
	return reshape(matrix, c, r)'
end
function clGetDeviceVendor(dev)
	local len = Array(Csize_t, 1)
	statusCheck(clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0, C_NULL, len))
	local info = Array(cl_char, len[1])
	statusCheck(clGetDeviceInfo(dev, CL_DEVICE_VENDOR, len[1], info, C_NULL))
	return bytestring(pointer(info))
end
function clGetDeviceName(dev)
	local len = Array(Csize_t, 1)
	statusCheck(clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, C_NULL, len))
	local info = Array(cl_char, len[1])
	statusCheck(clGetDeviceInfo(dev, CL_DEVICE_NAME, len[1], info, C_NULL))
	return bytestring(pointer(info))#equivalent to pointer(info)
end

function clGetFirstGPU()
	local props = vec(convert(Array{cl_context_properties, 2}, [CL_CONTEXT_PLATFORM 0 0]))
	local nplatform = Array(cl_uint, 1)
	#nplatform = cl_uint(0)
	statusCheck(clGetPlatformIDs(0,C_NULL,nplatform))
	#println(string("Found ",nplatform[1]," platforms!"))
	local platforms = Array(cl_platform_id, nplatform[1])
	statusCheck(clGetPlatformIDs(nplatform[1],platforms,C_NULL))
	
	#nplatform = cl_uint(0)
	for a in 1:nplatform[1]
		local platform = platforms[a]
		local ndev = Array(cl_uint, 1)
		ndev[1] = 0

		
		if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,0,C_NULL,ndev) == CL_SUCCESS)		##returns at least one gpu
			local devs = Array(cl_device_id, ndev[1]);
			statusCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,0,C_NULL,ndev))
			statusCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,ndev[1],devs,C_NULL));
			return devs[1]
		end
		##println(string("Found ",ndev[1]," devices!"))

		#println(string("Device Vendor: ", clGetDeviceName(devs[1])));
	end
	warn("CLBLAS Error:")
	Base.show_backtrace(Base.STDOUT,backtrace())
	println()
	throw(statusParse(CL_DEVICE_NOT_FOUND))
	return
end

function clGetGPUPlatform(device)
	local props = vec(convert(Array{cl_context_properties, 2}, [CL_CONTEXT_PLATFORM 0 0]))
	local nplatform = Array(cl_uint, 1)
	#nplatform = cl_uint(0)
	statusCheck(clGetPlatformIDs(0,C_NULL,nplatform))
	#println(string("Found ",nplatform[1]," platforms!"))
	local platforms = Array(cl_platform_id, nplatform[1])
	statusCheck(clGetPlatformIDs(nplatform[1],platforms,C_NULL))
	
	#nplatform = cl_uint(0)
	for a in 1:nplatform[1]
		local platform = platforms[a]
		local ndev = Array(cl_uint, 1)
		ndev[1] = 0

		
		if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,0,C_NULL,ndev) == CL_SUCCESS)		##returns at least one gpu
			local devs = Array(cl_device_id, ndev[1]);
			statusCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,0,C_NULL,ndev))
			statusCheck(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,ndev[1],devs,C_NULL));
			if (devs[1] == device)
				return platforms[a]
			end
		end
	end
	warn("CLBLAS Error:")
	Base.show_backtrace(Base.STDOUT,backtrace())
	println()
	throw(statusParse(CL_DEVICE_NOT_FOUND))
	return
end