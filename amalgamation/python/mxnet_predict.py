# coding: utf-8
# pylint: disable=invalid-name, too-many-arguments
"""Lightweight API for mxnet prediction.

This is for prediction only, use mxnet python package instead for most tasks.
"""
from __future__ import absolute_import

import datetime as dt

import os
import sys
import ctypes
import numpy as np

__all__ = ["Predictor", "load_ndarray_file"]

if sys.version_info[0] == 3:
    py_str = lambda x: x.decode('utf-8')
else:
    py_str = lambda x: x

def c_str(string):
    """"Convert a python string to C string."""
    return ctypes.c_char_p(string.encode('utf-8'))

def c_array(ctype, values):
    """Create ctypes array from a python array."""
    return (ctype * len(values))(*values)

def _find_lib_path():
    """Find mxnet library."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    api_path = os.path.join(curr_path, '../../lib/')
    dll_path = [curr_path, api_path]
    dll_path = [os.path.join(p, 'libmxnet_predict.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) == 0:
        raise RuntimeError('Cannot find the files.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))
    return lib_path


def _load_lib():
    """Load libary by searching possible path."""
    lib_path = _find_lib_path()
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    # DMatrix functions
    lib.MXGetLastError.restype = ctypes.c_char_p
    return lib


def _check_call(ret):
    """Check the return value of API."""
    if ret != 0:
        raise RuntimeError(py_str(_LIB.MXGetLastError()))

def calcTime(t0, t1):
    return t1-t0


_LIB = _load_lib()
# type definitions
mx_uint = ctypes.c_uint
mx_float = ctypes.c_float
mx_float_p = ctypes.POINTER(mx_float)
PredictorHandle = ctypes.c_void_p
NDListHandle = ctypes.c_void_p

devstr2type = {'cpu': 1, 'gpu': 2, 'cpu_pinned': 3}

class Predictor(object):
    """A predictor class that runs prediction.

    Parameters
    ----------
    symbol_json_str : str
        Path to the symbol file.

    param_raw_bytes : str, bytes
        The raw parameter bytes.

    input_shapes : dict of str to tuple
        The shape of input data

    dev_type : str, optional
        The device type of the predictor.

    dev_id : int, optional
        The device id of the predictor.
    """
    def __init__(self, symbol_file,
                 param_raw_bytes, input_shapes,
                 dev_type="cpu", dev_id=0):
        dev_type = devstr2type[dev_type]
        indptr = [0]
        sdata = []
        keys = []
        for k, v  in input_shapes.items():
            if not isinstance(v, tuple):
                raise ValueError("Expect input_shapes to be dict str->tuple")
            keys.append(c_str(k))
            sdata.extend(v)
            indptr.append(len(sdata))
        handle = PredictorHandle()
        param_raw_bytes = bytearray(param_raw_bytes)
        ptr = (ctypes.c_char * len(param_raw_bytes)).from_buffer(param_raw_bytes)
        
        t0 = dt.datetime.now()
        _check_call(_LIB.MXPredCreate(
            c_str(symbol_file),
            ptr, len(param_raw_bytes),
            ctypes.c_int(dev_type), ctypes.c_int(dev_id),
            mx_uint(len(indptr) - 1),
            c_array(ctypes.c_char_p, keys),
            c_array(mx_uint, indptr),
            c_array(mx_uint, sdata),
            ctypes.byref(handle)))
        
        self.tCreate = calcTime(t0, dt.datetime.now())

        self.handle = handle

    def __del__(self):
        _check_call(_LIB.MXPredFree(self.handle))

    def forward(self, **kwargs):
        """Perform forward to get the output.

        Parameters
        ----------
        **kwargs
            Keyword arguments of input variable name to data.

        Examples
        --------
        >>> predictor.forward(data=mydata)
        >>> out = predictor.get_output(0)
        """
        for k, v in kwargs.items():
            if not isinstance(v, np.ndarray):
                raise ValueError("Expect numpy ndarray as input")
            v = np.ascontiguousarray(v, dtype=np.float32)

            t0 = dt.datetime.now()
            _check_call(_LIB.MXPredSetInput(
                self.handle, c_str(k),
                v.ctypes.data_as(mx_float_p),
                mx_uint(v.size)))
            self.tSetInput = calcTime(t0, dt.datetime.now())

        t0 = dt.datetime.now()
        _check_call(_LIB.MXPredForward(self.handle))
        self.tPredFwd = calcTime(t0, dt.datetime.now())


    def get_output(self, index):
        """Get the index-th output.

        Parameters
        ----------
        index : int
            The index of output.

        Returns
        -------
        out : numpy array.
            The output array.
        """
        pdata = ctypes.POINTER(mx_uint)()
        ndim = mx_uint()
        _check_call(_LIB.MXPredGetOutputShape(
            self.handle, index,
            ctypes.byref(pdata),
            ctypes.byref(ndim)))
        shape = tuple(pdata[:ndim.value])
        data = np.empty(shape, dtype=np.float32)
        _check_call(_LIB.MXPredGetOutput(
            self.handle, mx_uint(index),
            data.ctypes.data_as(mx_float_p),
            mx_uint(data.size)))
        return data


def load_ndarray_file(nd_bytes):
    """Load ndarray file and return as list of numpy array.

    Parameters
    ----------
    nd_bytes : str or bytes
        The internal ndarray bytes

    Returns
    -------
    out : dict of str to numpy array or list of numpy array
        The output list or dict, depending on whether the saved type is list or dict.
    """
    handle = NDListHandle()
    olen = mx_uint()
    nd_bytes = bytearray(nd_bytes)
    ptr = (ctypes.c_char * len(nd_bytes)).from_buffer(nd_bytes)
    _check_call(_LIB.MXNDListCreate(
        ptr, len(nd_bytes),
        ctypes.byref(handle), ctypes.byref(olen)))
    keys = []
    arrs = []

    for i in range(olen.value):
        key = ctypes.c_char_p()
        cptr = mx_float_p()
        pdata = ctypes.POINTER(mx_uint)()
        ndim = mx_uint()
        _check_call(_LIB.MXNDListGet(
            handle, mx_uint(i), ctypes.byref(key),
            ctypes.byref(cptr), ctypes.byref(pdata), ctypes.byref(ndim)))
        shape = tuple(pdata[:ndim.value])
        dbuffer = (mx_float * np.prod(shape)).from_address(ctypes.addressof(cptr.contents))
        ret = np.frombuffer(dbuffer, dtype=np.float32).reshape(shape)
        ret = np.array(ret, dtype=np.float32)
        keys.append(py_str(key.value))
        arrs.append(ret)
    _check_call(_LIB.MXNDListFree(handle))

    if len(keys) == 0 or len(keys[0]) == 0:
        return arrs
    else:
        return {keys[i] : arrs[i] for i in range(len(keys))}


import pdb as pdb
pds = pdb.set_trace

def testForwarder(symbol_file, param_file, data_shape):

    import numpy as np

    inData = np.zeros(data_shape)
    print "data created"
    
    predictor = Predictor(open(symbol_file, "r").read(),
                          open(param_file, "rb").read(),
                          {'data':data_shape})

    predictor.forward(data=inData)

    print data_shape
    print ("CREATE=%5d,%5d  SET=%5d,%5d  FWD=%d5,%5d\n") % (predictor.tCreate.seconds,predictor.tCreate.microseconds,  predictor.tSetInput.seconds,predictor.tSetInput.microseconds,  predictor.tPredFwd.seconds,predictor.tPredFwd.microseconds)





if __name__ == '__main__':

    symbFile  = './models/Inception-BN-symbol.json'
    paramFile = './models/Inception-BN-0126.params'


    #t0 = dt.datetime.now()
    #time.sleep(10)
    #t1 = dt.datetime.now()
    #t = t1-t0
    #pds()

    for i in range(1, 10):
        data_shape = (2**i, 3, 224, 224)
        testForwarder(symbFile, paramFile, data_shape)
    

    #t1 = time.clock()
    #(t1 - t0)*1000.0 / iterations