# diopiTensor , diopiSize, diopiScalar, diopiReduction, diopiRoundMode, diopiError, TensorP, Context, Device, Dtype, \
#     diopi_tensor_copy_to_buffer, get_last_error_string, finalize_library, diopi_finalize = 1,2,3,4,5,6,7,8,9,10,11,12,13,14




diopiTensor = object
diopiSize = object
diopiScalar = object
diopiReduction = object
diopiRoundMode = object
diopiError = object
TensorP = object
Context = object
Dtype = object
class Dtype:
    float16 = float
    float32 = float
    float64 = float
    int32 = int
    int64 = int
    int16 = int 
    int8 = int
    uint8 = int
    complex64 = float
    complex128 = float
    bool = bool
class Device:
    Host = 1
    AIChip = 1
def diopi_tensor_copy_to_buffer(context, tensor, ptr):
    return 
def get_last_error_string():
    return
def finalize_library():
    return
def diopi_finalize():
    return
