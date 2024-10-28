import numpy as np
import pytest
import matplotlib
from mpi4py import MPI
import matplotlib.pyplot as plt
from LebwohlLasher import initdat, plotdat
from LebwohlLasher_cython import all_energy, get_order

# I recomend using command w to close the plot window after each test
# Run using:
# mpiexec -n 1 python -m pytest test_LebwohlLasher.py -v

@pytest.mark.parametrize("nmax", [15])
def test_plotdat(nmax):
    arr = initdat(nmax)
    # Test with pflag = 0 to check if it runs without error
    plotdat(arr, 0, nmax)
    # No assertion needed as we are just checking if it runs without error

@pytest.mark.parametrize("nmax", [15])
def test_plotdat(nmax):
    arr = initdat(nmax)
    # Test with pflag = 1 to check if it runs without error
    plotdat(arr, 1, nmax)
    # No assertion needed as we are just checking if it runs without error
    
@pytest.mark.parametrize("nmax", [15])
def test_plotdat(nmax):
    arr = initdat(nmax)
    # Test with pflag = 2 to check if it runs without error
    plotdat(arr, 2, nmax)
    # No assertion needed as we are just checking if it runs without error
    
@pytest.mark.parametrize("nmax", [15])
def test_plotdat(nmax):
    arr = initdat(nmax)
    # Test with pflag = 3 to check if it runs without error
    plotdat(arr, 3, nmax)
    # No assertion needed as we are just checking if it runs without error
    
@pytest.mark.parametrize("nmax", [5, 10, 15, 100, 500, 1000, 5000, 10000])
def test_all_energy(nmax):
    # All angles are 0, so energy should be 4 * nmax^2 
    arr = np.ones((nmax, nmax))
    energy = all_energy(arr, nmax, 1, 1, 1)
    # Expecting 4 time the number of elements in the array
    assert energy == -4 * nmax*nmax # Expected energy for a 10x10 array of ones
    
@pytest.mark.parametrize("nmax", [5, 10, 15, 100, 500, 1000, 5000, 10000])
def test_get_order(nmax):
    arr = np.ones((nmax, nmax))
    order = get_order(arr, nmax, 1, 1, 1)
    # Make sure its around 1 (small numerical errors can occur so use round)
    assert round(order,1) == 1.0 # Expected order for a 10x10 array of ones