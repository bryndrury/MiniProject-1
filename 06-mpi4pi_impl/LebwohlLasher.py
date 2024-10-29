import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI

def initdat(nmax) -> np.ndarray:
    if MPI.COMM_WORLD.Get_rank() != 0:
        return None
    
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi

def plotdat(arr, pflag, nmax):
    if MPI.COMM_WORLD.Get_rank() != 0:
        return None
    
    if pflag == 0:
        return
    
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))

    if pflag == 1:  # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i, j] = one_energy(arr, i, j, nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:  # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else: # colour the arrows in grayscale
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1 * nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax, comm_size) -> None:
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Number of processes: {:d}".format(comm_size),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
    
def one_energy(arr, ix, iy, nmax) -> float:
    energy = 0.0
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (ix + 1) % nmax
    iym = (ix - 1) % nmax

    angle = arr[ix,iy]-arr[ixp,iy]
    energy += 0.5*(1.0 - 3.0*np.cos(angle)**2)
    angle = arr[ix,iy]-arr[ixm,iy]
    energy += 0.5*(1.0 - 3.0*np.cos(angle)**2)
    angle = arr[ix,iy]-arr[ix,iyp]
    energy += 0.5*(1.0 - 3.0*np.cos(angle)**2)
    angle = arr[ix,iy]-arr[ix,iym]
    energy += 0.5*(1.0 - 3.0*np.cos(angle)**2)
    
    return energy
    
def all_energy(arr, nmax) -> float:
    energy = 0.0
    
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    start : int = int((rank/size) * nmax)
    if rank == size-1:
        end : int = nmax
    else:
        end : int = int(((rank+1)/size) * nmax)
    
    for i in range(start, end):
        for j in range(nmax):
            energy += one_energy(arr, i, j, nmax)
            
    energy = MPI.COMM_WORLD.allreduce(energy, op=MPI.SUM)
    
    return energy
                
def get_order(arr, nmax) -> float:
    if MPI.COMM_WORLD.Get_rank() != 0: # No need to waste energy on this calculation
        return 0.0
    
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    
    return eigenvalues.max()

def MC_step(arr, Ts, nmax) -> float:
    
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    start : int = int((rank/size) * nmax)
    if rank == size-1:
        end : int = nmax
    else:
        end : int = int(((rank+1)/size) * nmax)
    worksize = end - start
    
    scale = 0.1 + Ts
    accept = 0
    xran = np.random.randint(low=start, high=end, size=(worksize,nmax))
    yran = np.random.randint(low=0, high=nmax, size=(worksize,nmax))
    aran = np.random.normal(scale=scale, size=(worksize,nmax))
    
    for i in range(worksize):
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
                boltz = np.exp( -(energy1 - energy0) / Ts )
                if boltz >= np.random.uniform(0.0, 1.0):
                    accept += 1
                else:
                    arr[ix, iy] -= angle
                    
                    
    arr_np = np.asarray(arr[start:end, :])
    gathered_arr = MPI.COMM_WORLD.gather(arr_np, root=0)
    if rank == 0:
        new_arr = np.vstack(gathered_arr)
    else:
        new_arr = None
    new_arr = MPI.COMM_WORLD.bcast(new_arr, root=0)
    np.copyto(arr, new_arr) # copy the new array back to the original array
    reduced_accept = MPI.COMM_WORLD.reduce(accept, op=MPI.SUM, root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        return reduced_accept / (nmax * nmax)
    else:
        return None
        
def main(program, nsteps, nmax, temp, pflag):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Generate and broadcast the initial lattice
    lattice = initdat(nmax)
    lattice = comm.bcast(lattice, root=0)
    
    # Plot the initial lattice
    if rank == 0:
        plotdat(lattice, pflag, nmax)
    
    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)
    
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)
    
    initial = time.time()
    
    for step in range(nsteps+1):
        ratio[step] = MC_step(lattice, temp, nmax)
        energy[step] = all_energy(lattice, nmax)
        order[step] = get_order(lattice, nmax)
    
    final = time.time()
    runtime = final - initial
    
    if rank == 0:
        print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp}: Order: {order[nsteps-1]}, Time: {runtime} s")
        savedat(lattice,nsteps, temp, runtime, ratio, energy, order, nmax, size)
        plotdat(lattice, pflag, nmax)
    
    # print(f"Process {rank} is finalizing.")
    MPI.Finalize()
    
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) == 5:
        program = sys.argv[0]
        nsteps = int(sys.argv[1])
        nmax = int(sys.argv[2])
        temp = float(sys.argv[3])
        pflag = int(sys.argv[4])
    
        main(program, nsteps, nmax, temp, pflag)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))