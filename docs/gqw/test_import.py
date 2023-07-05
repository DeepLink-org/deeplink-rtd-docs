import ctypes
from ctypes import c_void_p
import numpy as np
import atexit
from export_runtime import diopiTensor, diopiSize, diopiScalar, diopiReduction, diopiRoundMode, diopiError, TensorP, Context, Device, Dtype, \
    diopi_tensor_copy_to_buffer, get_last_error_string, finalize_library, diopi_finalize

print("okayokayokay")