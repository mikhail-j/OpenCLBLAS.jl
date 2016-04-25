#=**
* Example of using high level single precision GEMM BLAS function.
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
include("sgemm.jl")

A = convert(Array{cl_float,2}, [[11, 12, 13, 14, 15]';[21, 22, 23, 24, 25]';[31, 32, 33, 34, 35]';[41, 42, 43, 44, 45]'])
B = convert(Array{cl_float,2}, [[11, 12, 13]';[21, 22, 23]';[31, 32, 33]';[41, 42, 43]';[51, 52, 53]'])
C = convert(Array{cl_float,2}, [[11, 12, 13]';[21, 22, 23]';[31, 32, 33]';[41, 42, 43]'])
OpenCLBLAS.sgemm!('N','N',cl_float(10),A,B, cl_float(20), C)
return C