# This file contains the DLPack API wrapped in Python style (see
# 'dlpack.h' for detail) and the utilities for Triton client to interact
# with DLPack
#
# Ref:
# https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
# https://github.com/dmlc/dlpack/blob/main/apps/numpy_dlpack/dlpack/from_numpy.py

import ctypes

from tritonclient.utils import raise_error

_c_str_dltensor = b"dltensor"

class DLDeviceType(ctypes.c_int):
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", DLDeviceType),
        ("device_id", ctypes.c_int),
    ]


class DLDataTypeCode(ctypes.c_uint8):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaquePointer = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("type_code", DLDataTypeCode),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]

class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]

class Context:
    def __init__(self, shape) -> None:
        # Convert the Python object to ctypes objects expected by
        # DLPack
        self.shape = (ctypes.c_int64 * len(shape))(*shape)
        # No strides: compact and row-major
        self.strides = ctypes.POINTER(ctypes.c_int64)()

    def _as_manager_ctx(self) -> ctypes.c_void_p:
        py_obj = ctypes.py_object(self)
        py_obj_ptr = ctypes.pointer(py_obj)
        ctypes.pythonapi.Py_IncRef(py_obj)
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(py_obj_ptr))
        return ctypes.cast(py_obj_ptr, ctypes.c_void_p)

@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _managed_tensor_deleter(handle: ctypes.c_void_p) -> None:
    dl_managed_tensor = DLManagedTensor.from_address(handle)
    py_obj_ptr = ctypes.cast(
        dl_managed_tensor.manager_ctx, ctypes.POINTER(ctypes.py_object)
    )
    py_obj = py_obj_ptr.contents
    ctypes.pythonapi.Py_DecRef(py_obj)
    ctypes.pythonapi.Py_DecRef(ctypes.py_object(py_obj_ptr))
    ctypes.pythonapi.PyMem_RawFree(handle)

@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _pycapsule_deleter(handle: ctypes.c_void_p) -> None:
    pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
        dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
            pycapsule, _c_str_dltensor
        )
        _managed_tensor_deleter(dl_managed_tensor)
        ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)

def triton_to_dlpack_dtype(dtype):
    if dtype == "BOOL":
        type_code = DLDataTypeCode.kDLBool
        bits = 1
    elif dtype == "INT8":
        type_code = DLDataTypeCode.kDLInt
        bits = 8
    elif dtype == "INT16":
        type_code = DLDataTypeCode.kDLInt
        bits = 16
    elif dtype == "INT32":
        type_code = DLDataTypeCode.kDLInt
        bits = 32
    elif dtype == "INT64":
        type_code = DLDataTypeCode.kDLInt
        bits = 64
    elif dtype == "UINT8":
        type_code = DLDataTypeCode.kDLUInt
        bits = 8
    elif dtype == "UINT16":
        type_code = DLDataTypeCode.kDLUInt
        bits = 16
    elif dtype == "UINT32":
        type_code = DLDataTypeCode.kDLUInt
        bits = 32
    elif dtype == "UINT64":
        type_code = DLDataTypeCode.kDLUInt
        bits = 64
    elif dtype == "FP16":
        type_code = DLDataTypeCode.kDLFloat
        bits = 16
    elif dtype == "FP32":
        type_code = DLDataTypeCode.kDLFloat
        bits = 32
    elif dtype == "FP64":
        type_code = DLDataTypeCode.kDLFloat
        bits = 64
    elif dtype == "BF16":
        type_code = DLDataTypeCode.kDLBfloat
        bits = 16
    elif dtype == "BYTES":
        raise_error("DLPack currently doesn't suppose BYTES type")
    else:
        raise_error("Can not covert unknown data type '{}' to DLPack data type".format(dtype))
    return DLDataType(type_code, bits, 1)