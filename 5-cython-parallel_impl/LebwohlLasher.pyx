import time
import datetime

from array import array
import random

import numpy as np
cimport numpy as np

cimport openmp
cimport cython
from cython.parallel cimport prange, parallel
from libc.math cimport cos, sin, M_PI, exp
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time as c_time

cdef extern from "omp.h":
    void omp_set_num_threads(int numthreads)

@cython.boundscheck(False)
cdef double one_energy(double[:, :] arr, int ix, int iy, int nmax) nogil:
    cdef double en = 0.0
    cdef double angle
    cdef int ixp, ixm, iyp, iym

    ixp = (ix + 1) % nmax
    ixm = (ix - 1 + nmax) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1 + nmax) % nmax

    angle = arr[ix, iy] - arr[ixp, iy]
    en += 0.5 * (1.0 - 3.0 * cos(angle)**2)
    
    angle = arr[ix, iy] - arr[ixm, iy]
    en += 0.5 * (1.0 - 3.0 * cos(angle)**2)
    
    angle = arr[ix, iy] - arr[ix, iyp]
    en += 0.5 * (1.0 - 3.0 * cos(angle)**2)
    
    angle = arr[ix, iy] - arr[ix, iym]
    en += 0.5 * (1.0 - 3.0 * cos(angle)**2)
    
    return en

def one_energy_py(arr, int ix, int iy, int nmax): # For python (other is for cython)
    return one_energy(np.asarray(arr), ix, iy, nmax)
    

@cython.boundscheck(False)
def all_energy(double[:, :]arr, int nmax, int numthreads):
    omp_set_num_threads(numthreads)
    
    cdef double energy = 0.0
    
    cdef int i, j
    
    for i in prange(nmax, nogil = True):
        for j in range(nmax):
            energy += one_energy(arr, i, j, nmax)
            
    return energy

@cython.boundscheck(False)
def get_order(double[:, :] arr, int nmax, int numthreads):
    omp_set_num_threads(numthreads)
    
    cdef double[:, :] Qab = np.zeros( (3, 3), dtype=np.float64)
    cdef double[:, :] delta = np.eye(3, dtype=np.float64)
    
    cdef double[:, :, :] lab = np.zeros( (3, nmax, nmax), dtype=np.float64)

    cdef int a, b, i, j
    
    for i in prange(nmax, nogil = True):
        for j in range(nmax):
            lab[0, i, j] = cos(arr[i, j])
            lab[1, i, j] = sin(arr[i, j])
            lab[2, i, j] = 0.5 * (3.0 * lab[0, i, j]**2 - 1.0)
    
    for a in range(3):
        for b in range(3):
            for i in prange(nmax, nogil = True):
                for j in range(nmax):
                    Qab[a,b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
    
    Qab = np.asarray(Qab) / (2 * nmax * nmax)
    
    eigenvalues, eigenvectors = np.linalg.eig(Qab)
    
    return eigenvalues.max()


@cython.boundscheck(False)
def MC_step(double[:, :] arr, double Ts, int nmax, int numthreads):
    omp_set_num_threads(numthreads)
    
    cdef double scale = 0.1 + Ts
    cdef double accept = 0
    
    cdef int[:, :] xran = np.random.randint(0, high=nmax, size=(nmax, nmax), dtype=np.int32)
    cdef int[:, :] yran = np.random.randint(0, high=nmax, size=(nmax, nmax), dtype=np.int32)
    cdef double[:, :] aran = np.random.normal(scale=scale, size=(nmax, nmax))
    
    cdef int i, j, ix, iy
    cdef double angle, energy0, energy1, boltz
    
    for i in prange(nmax, nogil = True):
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            angle = aran[i, j]
            
            energy0 = one_energy(arr, ix, iy, nmax)
            
            arr[ix, iy] += angle
            
            energy1 = one_energy(arr, ix, iy, nmax)
            
            if energy1 <= energy0:
                accept += 1
            else:
                boltz = exp( -(energy1 - energy0) / Ts )
                
                if boltz >= rand() / RAND_MAX:
                    accept += 1
                else:
                    arr[ix, iy] -= angle
    
    return accept / (nmax * nmax)
    