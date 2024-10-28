import cProfile
import pstats
import io
import numpy as np
from LebwohlLasher import main, all_energy, get_order  # Import your main function

def profile_code():
    # Arguments for the main function
    program = "LebwohlLasher_np.py"
    nsteps = 50
    nmax = 100
    temp = 0.5
    pflag = 0

    pr = cProfile.Profile()
    pr.enable()
    main(program, nsteps, nmax, temp, pflag)
    pr.disable()

    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

if __name__ == "__main__":
    profile_code()