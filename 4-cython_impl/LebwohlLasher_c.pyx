import time
import datetime

import numpy as np
cimport numpy as np

cimport cython
from libc.math cimport cos, sin, M_PI

# ==============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def savedat(np.ndarray[np.float64_t, ndim=2] arr, int nsteps, double Ts, double runtime, 
            np.ndarray[np.float64_t, ndim=1] energy, np.ndarray[np.float64_t, ndim=1] ratio,
            np.ndarray[np.float64_t, ndim=1] order, int nmax):
    
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()


# ==============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def initdat(int nmax):
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.zeros((nmax, nmax))
    cdef int i, j
    for i in range(nmax):
        for j in range(nmax):
            arr[i, j] = np.random.random_sample() * 2 * M_PI
    return arr


# ==============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def one_energy(np.ndarray[np.float64_t, ndim=2] arr, int ix, int iy, int nmax):
    cdef double en = 0.0
    
    cdef int ixp = (ix + 1) % nmax
    cdef int ixm = (ix - 1) % nmax
    cdef int iyp = (iy + 1) % nmax
    cdef int iym = (iy - 1) % nmax
    
    cdef np.float64_t ang 
    
    ang = arr[ix,iy] - arr[ixp,iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix,iy] - arr[ixm,iy]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix,iy] - arr[ix,iyp]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    ang = arr[ix,iy] - arr[ix,iym]
    en += 0.5 * (1.0 - 3.0 * cos(ang) ** 2)
    
    return en


# ==============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def all_energy(np.ndarray[np.float64_t, ndim=2] arr, int nmax):
    cdef double enall = 0.0
        
    cdef int i, j
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall


# ==============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def get_order(np.ndarray[np.float64_t, ndim=2] arr, int nmax):
    cdef np.ndarray[np.float64_t, ndim=2] Qab = np.zeros((3, 3))
    cdef np.ndarray[np.float64_t, ndim=2] delta = np.eye(3, 3)
    cdef np.ndarray[np.float64_t, ndim=3] lab = np.zeros((3, nmax, nmax))
    
    cdef int i, j, a, b
    
    for i in range(nmax):
        for j in range(nmax):
            lab[0, i, j] = cos(arr[i, j])
            lab[1, i, j] = sin(arr[i, j])
            lab[2, i, j] = 0.0

    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a, b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
                    
    Qab /= (2 * nmax * nmax)
    
    cdef np.ndarray[np.float64_t, ndim=1] eigenvalues 
    cdef np.ndarray[np.float64_t, ndim=2] eigenvectors
    
    eigenvalues, eigenvectors = np.linalg.eig(Qab)
    
    return eigenvalues.max()


# ==============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def MC_step(np.ndarray[np.float64_t, ndim=2] arr, double Ts, int nmax):
    cdef scale = 0.1 + Ts
    cdef accept = 0
    
    cdef float ang
    cdef np.ndarray[np.int64_t, ndim=2] xran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    cdef np.ndarray[np.int64_t, ndim=2] yran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    cdef np.ndarray[np.float64_t, ndim=2] aran = np.random.normal(scale=scale, size=(nmax, nmax))
    
    cdef int i, j
    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            if en1<=en0:
                accept += 1
            else:
                boltz = np.exp(-(en1 - en0) / Ts)
                
                if boltz >= np.random.uniform(0.0, 1.0):
                    accept += 1
                else:
                    arr[ix, iy] -= ang
                    
    return accept / (nmax * nmax)
    
# ==============================================================================
@cython.boundscheck(False)
@cython.wraparound(False) 
def main(str program, int nsteps, int nmax, double temp, int pflag):
    cdef np.ndarray[np.float64_t, ndim=2] lattice = initdat(nmax)
    cdef np.ndarray[np.float64_t, ndim=2] initial_lattice = lattice.copy()
    
    cdef np.ndarray[np.float64_t, ndim=1] energy = np.zeros(nsteps+1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] ratio = np.zeros(nsteps+1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] order = np.zeros(nsteps+1, dtype=np.float64)
    
    energy[0]   = all_energy(lattice, nmax)
    ratio[0]    = 0.5
    order[0]    = get_order(lattice, nmax)
    
    cdef double initial, final, runtime
    initial = time.time()
    
    cdef int iter
    for iter in range(1, nsteps+1):
        ratio[iter] = MC_step(lattice, temp, nmax)
        energy[iter] = all_energy(lattice, nmax)
        order[iter] = get_order(lattice, nmax)
    final = time.time()
    runtime = final - initial
    
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    savedat(lattice, nsteps, temp, runtime, energy, ratio, order, nmax)
    
    return initial_lattice, lattice