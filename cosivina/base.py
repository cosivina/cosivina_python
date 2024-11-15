import numpy as np
from logging import warning
import cosivina.options as options

if options.useNumba:
    from numba import njit, types, typed, prange
    from numba.experimental import jitclass

    intType = types.int32
    floatType = types.float64
    boolType = types.boolean
    stringType = types.unicode_type
    arrayType1D = types.Array(floatType, 1, 'C')
    arrayType2D = types.Array(floatType, 2, 'C')
    intArrayType = types.Array(intType, 1, 'C') # used e.g. for size parameters
    stringListType = types.ListType(stringType)
    sizeTupleType = types.UniTuple(types.int32, 2)
else:
    # define njit and jitclass to do nothing
    def njit(fn):
        return fn


    class jitclass:
        def __init__(self, spec):
            pass

        def __call__(self, fn):
            return fn

    intType = np.int32
    floatType = np.float64
    boolType = np.bool_
    stringType = str
    arrayType1D = np.ndarray
    arrayType2D = np.ndarray
    intArrayType = np.ndarray
    stringListType = list
    sizeTupleType = tuple



# note: currently using max() to determine init/step requirements
PS_FIXED = 0
PS_CHANGEABLE = 1
PS_INIT_REQUIRED = 2
PS_INIT_STEP_REQUIRED = 3
