#=**
* OpenCL type definitions and constants
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

#opencl type definitions (uses cuda's opencl/cl.h as reference)

typealias cl_bitfield Culong;
typealias cl_float Cfloat;
typealias cl_double Cdouble;
typealias cl_char Cchar;
typealias cl_uchar Cuchar;
typealias cl_short Cshort;
typealias cl_ushort Cushort;
typealias cl_int Cint;
typealias cl_uint Cuint;
typealias cl_long Clonglong;#64 bit long
typealias cl_ulong Culonglong;
typealias cl_half Float16;#cl_platform.h defines this as a unsigned short
#immutable cl_float2{T <: Float32} <: Number
immutable cl_float2
    s1::Float32
    s2::Float32
end
cl_float2{T <: Float32}(x::T) = cl_float2(x,Float32(0));
Base.convert(::Type{cl_float2}, x::Complex{Float32}) = cl_float2(real(x),imag(x));
Base.convert(::Type{Complex{Float32}}, f::cl_float2) = Complex{Float32}(f.s1,f.s2);
#possibly create a convert method for vector 1-d array => float2
immutable cl_double2
    s1::Float64
    s2::Float64
end
cl_double2{T <: Float64}(x::T) = cl_double2(x,Float64(0));
Base.convert(::Type{cl_double2}, x::Complex{Float64}) = cl_double2(real(x),imag(x));
Base.convert(::Type{Complex{Float64}}, f::cl_double2) = Complex{Float64}(f.s1,f.s2);
#possibly create a convert method for vector 1-d array => double2
typealias cl_device_id Ptr{Void};
typealias cl_platform_id Ptr{Void};
typealias cl_context Ptr{Void};
typealias cl_context_properties Cssize_t;
typealias cl_context_info cl_uint;
const CL_CONTEXT_REFERENCE_COUNT = cl_context_info(0x1080);
const CL_CONTEXT_DEVICES         = cl_context_info(0x1081);
const CL_CONTEXT_PROPERTIES      = cl_context_info(0x1082);
const CL_CONTEXT_NUM_DEVICES     = cl_context_info(0x1083);
const CL_CONTEXT_PLATFORM        = cl_context_info(0x1084);
typealias cl_mem Ptr{Void};
typealias cl_mem_info cl_uint;
const CL_MEM_TYPE                  = cl_mem_info(0x1100);
const CL_MEM_FLAGS                 = cl_mem_info(0x1101);
const CL_MEM_SIZE                  = cl_mem_info(0x1102);
const CL_MEM_HOST_PTR              = cl_mem_info(0x1103);
const CL_MEM_MAP_COUNT             = cl_mem_info(0x1104);
const CL_MEM_REFERENCE_COUNT       = cl_mem_info(0x1105);
const CL_MEM_CONTEXT               = cl_mem_info(0x1106);
const CL_MEM_ASSOCIATED_MEMOBJECT  = cl_mem_info(0x1107);
const CL_MEM_OFFSET                = cl_mem_info(0x1108);
typealias cl_event Ptr{Void};
typealias cl_event_info cl_uint;
const CL_EVENT_COMMAND_QUEUE            = cl_event_info(0x11D0);
const CL_EVENT_COMMAND_TYPE             = cl_event_info(0x11D1);
const CL_EVENT_REFERENCE_COUNT          = cl_event_info(0x11D2);
const CL_EVENT_COMMAND_EXECUTION_STATUS = cl_event_info(0x11D3);
const CL_EVENT_CONTEXT                  = cl_event_info(0x11D4);
typealias cl_command_queue Ptr{Void};
typealias cl_command_queue_properties cl_bitfield;
typealias cl_device_info cl_uint;
const CL_DEVICE_TYPE                           = cl_device_info(0x1000);
const CL_DEVICE_VENDOR_ID                      = cl_device_info(0x1001);
const CL_DEVICE_MAX_COMPUTE_UNITS              = cl_device_info(0x1002);
const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS       = cl_device_info(0x1003);
const CL_DEVICE_MAX_WORK_GROUP_SIZE            = cl_device_info(0x1004);
const CL_DEVICE_MAX_WORK_ITEM_SIZES            = cl_device_info(0x1005);
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR    = cl_device_info(0x1006);
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT   = cl_device_info(0x1007);
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT     = cl_device_info(0x1008);
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG    = cl_device_info(0x1009);
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT   = cl_device_info(0x100A);
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE  = cl_device_info(0x100B);
const CL_DEVICE_MAX_CLOCK_FREQUENCY            = cl_device_info(0x100C);
const CL_DEVICE_ADDRESS_BITS                   = cl_device_info(0x100D);
const CL_DEVICE_MAX_READ_IMAGE_ARGS            = cl_device_info(0x100E);
const CL_DEVICE_MAX_WRITE_IMAGE_ARGS           = cl_device_info(0x100F);
const CL_DEVICE_MAX_MEM_ALLOC_SIZE             = cl_device_info(0x1010);
const CL_DEVICE_IMAGE2D_MAX_WIDTH              = cl_device_info(0x1011);
const CL_DEVICE_IMAGE2D_MAX_HEIGHT             = cl_device_info(0x1012);
const CL_DEVICE_IMAGE3D_MAX_WIDTH              = cl_device_info(0x1013);
const CL_DEVICE_IMAGE3D_MAX_HEIGHT             = cl_device_info(0x1014);
const CL_DEVICE_IMAGE3D_MAX_DEPTH              = cl_device_info(0x1015);
const CL_DEVICE_IMAGE_SUPPORT                  = cl_device_info(0x1016);
const CL_DEVICE_MAX_PARAMETER_SIZE             = cl_device_info(0x1017);
const CL_DEVICE_MAX_SAMPLERS                   = cl_device_info(0x1018);
const CL_DEVICE_MEM_BASE_ADDR_ALIGN            = cl_device_info(0x1019);
const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE       = cl_device_info(0x101A);
const CL_DEVICE_SINGLE_FP_CONFIG               = cl_device_info(0x101B);
const CL_DEVICE_GLOBAL_MEM_CACHE_TYPE          = cl_device_info(0x101C);
const CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE      = cl_device_info(0x101D);
const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE          = cl_device_info(0x101E);
const CL_DEVICE_GLOBAL_MEM_SIZE                = cl_device_info(0x101F);
const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE       = cl_device_info(0x1020);
const CL_DEVICE_MAX_CONSTANT_ARGS              = cl_device_info(0x1021);
const CL_DEVICE_LOCAL_MEM_TYPE                 = cl_device_info(0x1022);
const CL_DEVICE_LOCAL_MEM_SIZE                 = cl_device_info(0x1023);
const CL_DEVICE_ERROR_CORRECTION_SUPPORT       = cl_device_info(0x1024);
const CL_DEVICE_PROFILING_TIMER_RESOLUTION     = cl_device_info(0x1025);
const CL_DEVICE_ENDIAN_LITTLE                  = cl_device_info(0x1026);
const CL_DEVICE_AVAILABLE                      = cl_device_info(0x1027);
const CL_DEVICE_COMPILER_AVAILABLE             = cl_device_info(0x1028);
const CL_DEVICE_EXECUTION_CAPABILITIES         = cl_device_info(0x1029);
const CL_DEVICE_QUEUE_PROPERTIES               = cl_device_info(0x102A);
const CL_DEVICE_NAME                           = cl_device_info(0x102B);
const CL_DEVICE_VENDOR                         = cl_device_info(0x102C);
const CL_DRIVER_VERSION                        = cl_device_info(0x102D);
const CL_DEVICE_PROFILE                        = cl_device_info(0x102E);
const CL_DEVICE_VERSION                        = cl_device_info(0x102F);
const CL_DEVICE_EXTENSIONS                     = cl_device_info(0x1030);
const CL_DEVICE_PLATFORM                       = cl_device_info(0x1031);
typealias cl_bool cl_uint;
const CL_FALSE = convert(cl_uint, 0);
const CL_TRUE = convert(cl_uint, 1);
typealias cl_mem_flags cl_bitfield;
const CL_MEM_READ_WRITE = cl_bitfield(1 << 0);
const CL_MEM_WRITE_ONLY = cl_bitfield(1 << 1);
const CL_MEM_READ_ONLY  = cl_bitfield(1 << 2);
const CL_MEM_USE_HOST_PTR = cl_bitfield(1 << 3);
const CL_MEM_ALLOC_HOST_PTR = cl_bitfield(1 << 4);
const CL_MEM_COPY_HOST_PTR = cl_bitfield(1 << 5);
typealias cl_device_type cl_bitfield;
const CL_DEVICE_TYPE_DEFAULT = cl_bitfield(1 << 0);
const CL_DEVICE_TYPE_CPU = cl_bitfield(1 << 1);
const CL_DEVICE_TYPE_GPU = cl_bitfield(1 << 2);
const CL_DEVICE_TYPE_ACCELERATOR = cl_bitfield(1 << 3);
const CL_DEVICE_TYPE_ALL = cl_bitfield(0xFFFFFFFF);

#===============  OpenCL\cl.h definitions ===============#
const CL_SUCCESS                                   = 0;
const CL_DEVICE_NOT_FOUND                          = -1;
const CL_DEVICE_NOT_AVAILABLE                      = -2;
const CL_COMPILER_NOT_AVAILABLE                    = -3;
const CL_MEM_OBJECT_ALLOCATION_FAILURE             = -4;
const CL_OUT_OF_RESOURCES                          = -5;
const CL_OUT_OF_HOST_MEMORY                        = -6;
const CL_PROFILING_INFO_NOT_AVAILABLE              = -7;
const CL_MEM_COPY_OVERLAP                          = -8;
const CL_IMAGE_FORMAT_MISMATCH                     = -9;
const CL_IMAGE_FORMAT_NOT_SUPPORTED                = -10;
const CL_BUILD_PROGRAM_FAILURE                     = -11;
const CL_MAP_FAILURE                               = -12;
const CL_MISALIGNED_SUB_BUFFER_OFFSET              = -13;
const CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14;
const CL_INVALID_VALUE                             = -30;
const CL_INVALID_DEVICE_TYPE                       = -31;
const CL_INVALID_PLATFORM                          = -32;
const CL_INVALID_DEVICE                            = -33;
const CL_INVALID_CONTEXT                           = -34;
const CL_INVALID_QUEUE_PROPERTIES                  = -35;
const CL_INVALID_COMMAND_QUEUE                     = -36;
const CL_INVALID_HOST_PTR                          = -37;
const CL_INVALID_MEM_OBJECT                        = -38;
const CL_INVALID_IMAGE_FORMAT_DESCRIPTOR           = -39;
const CL_INVALID_IMAGE_SIZE                        = -40;
const CL_INVALID_SAMPLER                           = -41;
const CL_INVALID_BINARY                            = -42;
const CL_INVALID_BUILD_OPTIONS                     = -43;
const CL_INVALID_PROGRAM                           = -44;
const CL_INVALID_PROGRAM_EXECUTABLE                = -45;
const CL_INVALID_KERNEL_NAME                       = -46;
const CL_INVALID_KERNEL_DEFINITION                 = -47;
const CL_INVALID_KERNEL                            = -48;
const CL_INVALID_ARG_INDEX                         = -49;
const CL_INVALID_ARG_VALUE                         = -50;
const CL_INVALID_ARG_SIZE                          = -51;
const CL_INVALID_KERNEL_ARGS                       = -52;
const CL_INVALID_WORK_DIMENSION                    = -53;
const CL_INVALID_WORK_GROUP_SIZE                   = -54;
const CL_INVALID_WORK_ITEM_SIZE                    = -55;
const CL_INVALID_GLOBAL_OFFSET                     = -56;
const CL_INVALID_EVENT_WAIT_LIST                   = -57;
const CL_INVALID_EVENT                             = -58;
const CL_INVALID_OPERATION                         = -59;
const CL_INVALID_GL_OBJECT                         = -60;
const CL_INVALID_BUFFER_SIZE                       = -61;
const CL_INVALID_MIP_LEVEL                         = -62;
const CL_INVALID_GLOBAL_WORK_SIZE                  = -63;
const CL_INVALID_PROPERTY                          = -64;