# Copyright (c) 2023, DeepLink.
import os
from enum import Enum, unique
from ctypes import (CDLL, RTLD_GLOBAL, cdll, byref, Structure, Union, POINTER)
from ctypes import (c_void_p, c_char_p, c_int64, c_int32, c_double)
from .dtype import Dtype
import numpy as np
import atexit


@unique
class Device(Enum):
    Host = 0
    AIChip = 1


def device(dev: str) -> Device:
    if dev == "cpu" or dev == "host":
        return Device(0)
    else:
        return Device(1)


def from_numpy_dtype(dtype: np.dtype) -> Dtype:
    if dtype == np.int8:
        return Dtype.int8
    elif dtype == np.int16:
        return Dtype.int16
    elif dtype == np.int32:
        return Dtype.int32
    elif dtype == np.int64:
        return Dtype.int64
    elif dtype == np.uint8:
        return Dtype.uint8
    elif dtype == np.uint16:
        return Dtype.uint16
    elif dtype == np.uint32:
        return Dtype.uint32
    elif dtype == np.uint64:
        return Dtype.uint64
    elif dtype == np.float16:
        return Dtype.float16
    elif dtype == np.float32:
        return Dtype.float32
    elif dtype == np.float64:
        return Dtype.float64
    elif dtype == np.bool_:
        return Dtype.bool
    elif dtype == np.complex64:
        return Dtype.complex64
    elif dtype == np.complex128:
        return Dtype.complex128
    else:
        return None


def to_numpy_dtype(dtype: Dtype) -> np.dtype:
    if dtype == Dtype.int8:
        return np.int8
    elif dtype == Dtype.int16:
        return np.int16
    elif dtype == Dtype.int32:
        return np.int32
    elif dtype == Dtype.int64:
        return np.int64
    elif dtype == Dtype.uint8:
        return np.uint8
    elif dtype == Dtype.uint16:
        return np.uint16
    elif dtype == Dtype.uint32:
        return np.uint32
    elif dtype == Dtype.uint64:
        return np.uint64
    elif dtype == Dtype.float16:
        return np.float16
    elif dtype == Dtype.float32:
        return np.float32
    elif dtype == Dtype.float64:
        return np.float64
    elif dtype == Dtype.bool:
        return np.bool_
    elif dtype == Dtype.complex64:
        return np.complex64
    elif dtype == Dtype.complex128:
        return np.complex128
    else:
        return None


def compute_nhwc_stride_2d(sizes, itemsize=1):
    dim = len(sizes)
    strides = [itemsize for i in range(dim)]
    assert dim == 3 or dim == 4, "not supported dim"
    if dim == 3:
        strides[0] = itemsize
        strides[2] = strides[0] * sizes[0]
        strides[1] = strides[2] * sizes[2]
    elif dim == 4:
        strides[1] = itemsize
        strides[3] = strides[0] * sizes[1]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    return strides


def compute_nhwc_stride_3d(sizes, itemsize=1):
    dim = len(sizes)
    strides = [itemsize for i in range(dim)]
    assert dim == 4 or dim == 5, "not supported dim"
    if dim == 4:
        strides[0] = itemsize
        strides[3] = strides[0] * sizes[0]
        strides[2] = strides[3] * sizes[3]
        strides[1] = strides[2] * sizes[2]
    elif dim == 5:
        strides[1] = itemsize
        strides[4] = strides[0] * sizes[1]
        strides[3] = strides[4] * sizes[4]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    return strides


def compute_nhwc_stride(size, itemsize=1, name=None):
    if name == '2d':
        return compute_nhwc_stride_2d(size, itemsize)
    if name == '3d':
        return compute_nhwc_stride_3d(size, itemsize)

    dim = len(size)
    if dim < 5:
        return compute_nhwc_stride_2d(size, itemsize)
    else:
        return compute_nhwc_stride_3d(size, itemsize)


_cur_dir = os.path.dirname(os.path.abspath(__file__))

diopirt_lib = object

device_impl_lib = object


def on_diopi_rt_exit():
    pass


atexit.register(on_diopi_rt_exit)


def get_last_error():
    pass


ContextHandle = c_void_p
TensorHandle = c_void_p


class Context:

    def __init__(self):
        pass

    def __del__(self):
        pass

    def get_handle(self):
        pass

    def clear_tensors(self):
        pass


default_context = Context()


class Sizes(Structure):
    _fields_ = [("data", POINTER(c_int64)), ("len", c_int64)]

    def __init__(self, shape=()):
        pass


class ScalarUnion(Union):
    _fields_ = [("fval", c_double), ("ival", c_int64)]


class Scalar(Structure):
    _fields_ = [("stype", c_int32), ("val", ScalarUnion)]

    def __init__(self, value, dtype=None):
        pass


class Tensor:
    def __init__(
        self,
        size,
        dtype,
        stride=None,
        context_handle=default_context.get_handle(),
        tensor_handle=None,
    ):
        pass

    @classmethod
    def from_handle(cls, tensor_handle):
        pass

    def __str__(self):
        pass

    def raw_like(self):
        pass

    def numel(self):
        pass

    def size(self):
        pass

    def shape(self):
        pass

    def get_stride(self):
        pass

    def itemsize(self):
        pass

    def get_device(self):
        pass

    def get_dtype(self):
        pass

    def reset_shape(self, shape):
        pass

    @classmethod
    def from_numpy(cls, darray):
        pass

    def numpy(self) -> np.ndarray:
        pass


def raw_like(tensor) -> Tensor:
    pass
