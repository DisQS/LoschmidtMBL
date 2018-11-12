import os
import math
import cmath
import operator
import itertools
import functools
from numbers import Integral
import numpy as np
from numpy.matlib import zeros
import scipy.sparse as sp
from cytoolz import partition_all, unique
import numba
from numba import njit

def make_immutable(mat):
    """Make array not writable, so that it can be stored in cache safely.
    Args:
        mat(sparse or dense array of floats)            = Matrix to make immutable
    Returns:
        Non-writable mat
    """
    if checksparse(mat):
        mat.data.flags.writeable = False
        if mat.format in {'csr', 'csc', 'bsr'}:
            mat.indices.flags.writeable = False
            mat.indptr.flags.writeable = False
        elif mat.format == 'coo':
            mat.row.flags.writeable = False
            mat.col.flags.writeable = False
    else:
        mat.flags.writeable = False

def checksparse(obj):
    """Checks if obj is sparse.
    """
    return isinstance(obj, sp.spmatrix)

def checkdense(obj):
    """Checks if obj is dense.
    """
    return isinstance(obj, np.ndarray)

def checkreal(obj, **allclose_opts):
    """Checks if obj is approximately real.
    """
    data = obj.data if checksparse(obj) else obj

    # check dtype
    if np.checkrealobj(data):
        return True

    # else check explicitly
    return np.allclose(data.imag, 0.0, **allclose_opts)

def checkop(obj):
    """Checks if obj is an operator
    """
    s = obj.shape
    return len(s) == 2 and (s[0] > 1) and (s[1] > 1)

def checkherm(obj, **allclose_opts):
    """Checks if obj is hermitian.

    Parameters
    ----------
    obj : dense or sparse operator
        Matrix to check.

    Returns
    -------
    bool
    """
    if checksparse(obj):
        return allclose_sparse(obj, dag(obj), **allclose_opts)
    else:
        return np.allclose(obj, dag(obj), **allclose_opts)

def dag(obj):
    """Hermitian conjugate transpose
    """
    try:
        return obj.T
    except AttributeError:
        return obj.conj().T

#---------------------------------------------------#
#   Matrix and vector operations                    #
#---------------------------------------------------#
def dot(a, b):
    """Matrix multiplication
    Args:
        a(1d or 2d array)                           = Dense or sparse operator
        b(1d or 2d array)                           = Dense or sparse operator
    Returns:
        Dot product of a and b.
    """
    return a @ b

def vdot(a, b):
    """Hermitian inner product (dot with complex conjugate)
    """
    return np.vdot(a, b)

@njit
def prod_dense(a, b):
    """Element-wise product of two operators with numba
    Args:
        a(2d array)                                 = Dense operator
        b(2d array)                                 = Dense operator
    Returns:
        a*b                                         = Element-wise product of a and b
    """
    return a*b

def prod(a, b):
    if checksparse(a):
        return a.multiply(b)
    elif checksparse(b):
        return b.multiply(a)
    else:
        return prod_dense(a,b)

#---------------------------------------------------#
#   Kronecker product                               #
#---------------------------------------------------#
@njit(parallel=True)
def kron_dense(a, b, threshold=4096):
    """Kronecker product of two dense operators (parallelized for large sizes)
    Args:
        a,b (2d array)                              = Dense operators
    Returns:
        kron (2d array)                             = Kronecker product of a and b
    """
    i,j = a.shape
    k,l = b.shape

    kron = np.empty((i*k, j*l), dtype = np.float64)

    if kron.size > threshold:
        for m in numba.prange(i):
            for n in range(j):
                mm, fm = m * k, (m + 1) * k
                mn, fn = n * l, (n + 1) * l
                kron[mm:fm, mn:fn] = a[m, n] * b
        return kron
    else:
        for m in range(i):
            for n in range(j):
                mm, fm = m * k, (m + 1) * k
                mn, fn = n * l, (n + 1) * l
                kron[mm:fm, mn:fn] = a[m, n] * b
        return kron

def kron(a,b):
    """Kronecker product of two generic operators (dense or sparse)
    Args:
        a,b (2d array)                              = Generic operators
    Returns:
        kron (2d array)                             = Kronecker product of a and b
    """
    if checksparse(a) or checksparse(b):
        return sp.kron(a,b)
    else:
        return kron_dense(a,b)

#---------------------------------------------------#
#   Expectation values                              #
#---------------------------------------------------#

def expec(a,b):
    """Expectation value of an operator acting on a state vector
    Args:
        a(1d or 2d array)                           = First vector or operator
        b(1d or 2d array)                           = Second vector or operator
    Returns:
        out(float)                                  = - for two vector it is |<a|b><b|a>|
                                                      - for an operator and a vector it is <a|b|a>
                                                      - for two operators is the Hilbert-Schmidt inner product
    """
    if checkop(a) and not checkop(b):
        if checksparse(a):
            return (dag(b) @ (a @ b))
        return vdot(b, a @ b)
    elif checkop(b) and not checkop(a):
        if checksparse(b):
            return (dag(a) @ (b @ a))
        return vdot(a, b @ a)
    elif not checkop(a) and not checkop(b):
        return np.absolute(vdot(a,b))**2
    elif checkop(a) and checkop(b):
        if checksparse(a) or checksparse(b):
            A = dot(a,b)
            return np.sum(A.diagonal())
        return np.trace(a @ b)
