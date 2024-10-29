import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from LebwohlLasher_pyx import LebwohlLasher, all_energy, get_order, MC_step, one_energy_py

def initdat(nmax):
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    return arr    

def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return
    
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))
    if pflag == 1:
        mpl.rc("image", cmap="rainbow")
        for i in range(nmax):
            for j in range(nmax):
                cols[i, j] = one_energy_py(arr, i, j, nmax)
        norm = mpl.colors.Normalize(cols.min(), cols.max())
    
    elif pflag == 2:
        mpl.rc("image", cmap="hsv")
        cols = arr%np.pi
        norm = mpl.colors.Normalize(0, 2*np.pi)
    
    else:
        mpl.rc("image", cmap="gist_gray")
        cols = np.zeros_like(arr)
        norm = mpl.colors.Normalize(0, 2*np.pi)
        
    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y , u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
    
def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax, numthreads):
    current_datetime  = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = f"Python-LL-Output-{current_datetime}.txt"
    FileOut = open(filename, "w")
    
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Number of threads:   {:d}".format(numthreads),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()

def main(program, nsteps, nmax, temp, pflag, numthreads):
        
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)
    
    energy  = np.zeros(nsteps+1)
    ratio   = np.zeros(nsteps+1)
    order   = np.zeros(nsteps+1)
    
    energy[0]   = all_energy(lattice, nmax, numthreads)
    ratio[0]    = 0.5
    order[0]    = get_order(lattice, nmax, numthreads)
    
    initial = time.time()
    
    for iter in range(1, nsteps+1):
        print(f" Step: {iter}", end="\r")
        
        ratio[iter] = MC_step(lattice, temp, nmax, numthreads)
        energy[iter] = all_energy(lattice, nmax, numthreads)
        order[iter] = get_order(lattice, nmax, numthreads)
        
    final = time.time()
    runtime = final - initial
    
    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp}, Order: {order[nsteps-1]:.2f}, Thread Count: {numthreads}, Time: {runtime} s")
    
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax, numthreads)
    plotdat(lattice, pflag, nmax)


if __name__ == '__main__':
    if int(len(sys.argv)) == 7:
        PROGRAM = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        NUMTHREADS = int(sys.argv[5])
        ABSTRACT = int(sys.argv[6])
        
        if ABSTRACT:
            print("Running Lebwohl-Lasher model with Cython Abstraction")
            init_arr, arr = LebwohlLasher(ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, NUMTHREADS)
        else:
            print("Running Lebwohl-Lasher model with Python")
            main(PROGRAM, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, NUMTHREADS)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADCOUNT> <ABSTRACTION>")