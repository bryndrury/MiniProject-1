"""
Run at the command line by typing:

python LebwohlLasher_np.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
    ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
        has attempted a change once on average (i.e. SIZE*SIZE attempts)
    SIZE = side length of square lattice
    TEMPERATURE = reduced temperature in range 0.0 - 2.0.
    PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
"""

import sys
import time
import datetime
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib as mpl

def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
        program (string) -- name of the program
        nsteps (int) -- number of Monte Carlo steps
        nmax (int) -- side length of square lattice
        temo (float) -- reduced temperature in range 0.0 - 2.0
        pflag (int) -- 0 for no plot, 1 for energy plot, 2 for angle plot
        
    Description:
        Main function to run the Lebwohl-Lasher model.
    
    Returns:
        NULL
    """
    
    lattice = initdat(nmax)
    
    energy = np.zeros(nsteps+1, dtype=np.dtype)
    ratio = np.zeros(nsteps+1, dtype=np.dtype)
    order = np.zeros(nsteps+1, dtype=np.dtype)
    
    plotdat(lattice, pflag, nmax)
    
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)
    
    initial = time.time()
    for iter in range(1, nsteps+1):
        ratio[iter]     = MC_step(lattice, temp, nmax)
        energy[iter]    = all_energy(lattice, nmax)
        order[iter]     = get_order(lattice, nmax)
    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)
    

def initdat(nmax):
    arr = np.random.random_sample((nmax, nmax))
    arr = np.vectorize(lambda x: x * 2 * np.pi)(arr)
    return arr


def all_energy(arr, nmax):
    arr_right       = np.roll(arr, -1, axis=1)
    arr_left        = np.roll(arr,  1, axis=1)
    arr_up          = np.roll(arr, -1, axis=0)
    arr_down        = np.roll(arr,  1, axis=0)
    
    angles_right    = arr - arr_right
    angles_left     = arr - arr_left
    angles_up       = arr - arr_up
    angles_down     = arr - arr_down
    
    angles = np.array([arr - arr_right, arr - arr_left, arr - arr_up, arr - arr_down])
    energy = 0.5 * (1.0 - 3.0 * np.cos(angles) ** 2)
    
    total_energy = np.sum(energy)
    
    return total_energy


def get_order(arr, nmax):
    delta = np.eye(3)

    lab = np.array([np.cos(arr), np.sin(arr), np.zeros_like(arr)])

    Qab = np.einsum('ai,bi->ab', lab.reshape(3,-1), lab.reshape(3,-1)) * 3 - delta * nmax * nmax
    Qab /= (2 * nmax * nmax)    

    eigenvalues = np.linalg.eigvalsh(Qab)
    return eigenvalues.max()


@nb.njit
def one_energy(arr,ix,iy,nmax):
    en = 0.0
    ixp = (ix+1) % nmax # These are the coordinates
    ixm = (ix-1) % nmax # of the neighbours
    iyp = (iy+1) % nmax # with wraparound
    iym = (iy-1) % nmax # 
    
    neighbours = [(ixp, iy), (ixm, iy), (ix, iyp), (ix, iym)]
    angles = arr[ix, iy] - np.array([arr[nx, ny] for nx, ny in neighbours])
    en = 0.5 * np.sum(1.0 - 3.0 * np.cos(angles)**2)
    
    return en


def MC_step(arr, Ts, nmax):
    scale = 0.1 + Ts
    accept = 0

    # Pre-compute random numbers
    xran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    yran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    aran = np.random.normal(scale=scale, size=(nmax, nmax))

    # Compute initial energies
    en0 = np.zeros((nmax, nmax))
    for i in range(nmax):
        for j in range(nmax):
            en0[i, j] = one_energy(arr, xran[i, j], yran[i, j], nmax)

    # Apply angle changes
    arr[xran, yran] += aran

    # Compute new energies
    en1 = np.zeros((nmax, nmax))
    for i in range(nmax):
        for j in range(nmax):
            en1[i, j] = one_energy(arr, xran[i, j], yran[i, j], nmax)

    # Calculate acceptance
    delta_en = en1 - en0
    accept_mask = (delta_en <= 0) | (np.exp(-delta_en / Ts) >= np.random.uniform(0.0, 1.0, size=(nmax, nmax)))
    accept = np.sum(accept_mask)

    # Revert changes where not accepted
    arr[xran[~accept_mask], yran[~accept_mask]] -= aran[~accept_mask]

    return accept / (nmax * nmax)


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
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
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
    
    
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    # Create filename based on current date and time.
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

# ============================================================================== #
# ==                        Entry point of the program                        == #
# ============================================================================== #
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME    : str   = sys.argv[0]
        ITERATIONS  : int   = int(sys.argv[1])
        SIZE        : int   = int(sys.argv[2])
        TEMPERATURE : float = float(sys.argv[3])
        PLOTFLAG    : bool  = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
        
