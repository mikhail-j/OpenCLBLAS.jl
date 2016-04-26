#=**
* OpenCL BLAS type definitions and constants
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

include("cl_typedef.jl");

typealias clblasFloatComplex Complex{Float32};#technically this is a cl_float2
typealias clblasDoubleComplex Complex{Float64};#technically this is a cl_float2
typealias clblasOrder Cint;
const clblasRowMajor = Cint(0);           #/**< Every row is placed sequentially */
const clblasColumnMajor = Cint(1);         #/**< Every column is placed sequentially */
typealias clblasTranspose Cint;
const clblasNoTrans = Cint(0);                #/**< Operate with the matrix. */
const clblasTrans = Cint(1);                  #/**< Operate with the transpose of the matrix. */
const clblasConjTrans = Cint(2);              #/**< Operate with the conjugate transpose of the matrix. */
#=
/** Used by the Hermitian, symmetric and triangular matrix
 * routines to specify whether the upper or lower triangle is being referenced.
 */
 =#
typealias clblasUplo Cint;
const clblasUpper = Cint(0);          #/**< Upper triangle. */
const clblasLower = Cint(1);          #/**< Lower triangle. */
#=
/** It is used by the triangular matrix routines to specify whether the
 * matrix is unit triangular.
 */
=#
typealias clblasDiag Cint;
const clblasUnit       = Cint(0);        #/**< Unit triangular. */
const clblasNonUnit    = Cint(1);        #/**< Non-unit triangular. */

#/** Indicates the side matrix A is located relative to matrix B during multiplication. */
typealias clblasSide Cint;
const clblasLeft     = Cint(0);    #/**< Multiply general matrix by symmetric, Hermitian or triangular matrix on the left. */
const clblasRight    = Cint(1);    #/**< Multiply general matrix by symmetric, Hermitian or triangular matrix on the right. */
typealias clblasStatus Cint;
const clblasSuccess = 0;#                        = CL_SUCCESS,
const clblasInvalidValue = -30;#                 = CL_INVALID_VALUE,
const clblasInvalidCommandQueue = -36;#          = CL_INVALID_COMMAND_QUEUE,
const clblasInvalidContext =-34;#                 = CL_INVALID_CONTEXT,
const clblasInvalidMemObject =-38;#               = CL_INVALID_MEM_OBJECT,
const clblasInvalidDevice = -33;#                  = CL_INVALID_DEVICE,
const clblasInvalidEventWaitList = -57;#           = CL_INVALID_EVENT_WAIT_LIST,
const clblasOutOfResources = -5;#                 = CL_OUT_OF_RESOURCES,
const clblasOutOfHostMemory = -6;#                = CL_OUT_OF_HOST_MEMORY,
const clblasInvalidOperation = -59;#               = CL_INVALID_OPERATION,
const clblasCompilerNotAvailable = -3;#           = CL_COMPILER_NOT_AVAILABLE,
const clblasBuildProgramFailure = -11;#            = CL_BUILD_PROGRAM_FAILURE
const clblasNotImplemented = -1024;
const clblasNotInitialized = -1023;#               not sure if it iterates like this
const clblasInvalidMatA            = -1022;#       /**< Matrix A is not a valid memory object */
const clblasInvalidMatB            = -1021;#       /**< Matrix B is not a valid memory object */
const clblasInvalidMatC            = -1020;#       /**< Matrix C is not a valid memory object */
const clblasInvalidVecX            = -1019;#       /**< Vector X is not a valid memory object */
const clblasInvalidVecY            = -1018;#       /**< Vector Y is not a valid memory object */
const clblasInvalidDim             = -1017;#       /**< An input dimension (M,N,K) is invalid */
const clblasInvalidLeadDimA        = -1016;#       /**< Leading dimension A must not be less than the size of the first dimension */
const clblasInvalidLeadDimB        = -1015;#       /**< Leading dimension B must not be less than the size of the second dimension */
const clblasInvalidLeadDimC        = -1014;#       /**< Leading dimension C must not be less than the size of the third dimension */
const clblasInvalidIncX            = -1013;#       /**< The increment for a vector X must not be 0 */
const clblasInvalidIncY            = -1012;#       /**< The increment for a vector Y must not be 0 */
const clblasInsufficientMemMatA    = -1011;#       /**< The memory object for Matrix A is too small */
const clblasInsufficientMemMatB    = -1010;#       /**< The memory object for Matrix B is too small */
const clblasInsufficientMemMatC    = -1009;#       /**< The memory object for Matrix C is too small */
const clblasInsufficientMemVecX    = -1008;#       /**< The memory object for Vector X is too small */
const clblasInsufficientMemVecY    = -1007;#       /**< The memory object for Vector Y is too small */