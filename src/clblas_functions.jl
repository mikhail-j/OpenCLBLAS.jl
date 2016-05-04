#=**
* AMD OpenCL clBLAS functions from "https://github.com/clMathLibraries/clBLAS/blob/master/src/clBLAS.h"
* Functions such as clblasSetup or clblasTeardown
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

include("clblas_typedef.jl")

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
		Ptr{cl_command_queue},
		cl_uint,
		#Ref{cl_event},
		#AMD's OpenCL driver (Windows 7 x64) throws invalid event if argument type is Ref{cl_event}
		Ptr{cl_event},
		Ref{cl_event}),
		o,tA,tB,M,N,K,alpha,A,offA,lda,B,offB,ldb,beta,C,offC,ldc,ncq,cq,ne,wle,e)
end