import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from LebwohlLasher import LebwohlLasher, energies, initdat, all_energy, get_order, MC_step

def plotdat(arr,pflag,nmax):
    if pflag==0:
        return 
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        cols = np.array(energies(arr, nmax)).reshape(nmax,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = np.array(arr)%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
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
    
def main(program, nsteps, nmax, temp, pflag, nthreads):
    
    lattice = initdat(nmax, nthreads)
    
    plotdat(lattice, pflag, nmax)
    
    energy  = np.zeros(nsteps+1)
    ratio   = np.zeros(nsteps+1)
    order   = np.zeros(nsteps+1)
    
    energy[0]   = all_energy(lattice, nmax, nthreads)
    ratio[0]    = 0.5
    order[0]    = get_order(lattice, nmax, nthreads)
    
    inital = time.time()
    
    for iter in range(1, nsteps+1):
        print(f"Iteration: {iter}", end="\r")
        
        ratio[iter] = MC_step(lattice, temp, nmax, nthreads)
        energy[iter] = all_energy(lattice, nmax, nthreads)
        order[iter] = get_order(lattice, nmax, nthreads)
        
    finial = time.time()
    runtime = finial - inital
    
    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp}, Order: {order[nsteps-1]:.2f}, Thread Count: {nthreads}, Time: {runtime} s")
    
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax, nthreads)
    plotdat(lattice, pflag, nmax)
        
if __name__ == '__main__':
    if int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PFLAG = int(sys.argv[4])
        THREADCOUNT = int(sys.argv[5])
        
        # Using a similar method to the original code to run the simulation
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PFLAG, THREADCOUNT)
        
        # Abstracting the code behind a function (cpp implementation) for a greater speedup
        initial_state, final_state = LebwohlLasher(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, THREADCOUNT)
        plotdat(initial_state, PFLAG, SIZE)
        plotdat(final_state, PFLAG, SIZE)
        
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADCOUNT>".format(sys.argv[0]))
#=======================================================================