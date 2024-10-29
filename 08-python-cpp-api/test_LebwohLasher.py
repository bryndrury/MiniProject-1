import numpy as np
import pytest
import matplotlib.pyplot as plt
from LebwohlLasher_py import plotdat
from LebwohlLasher import initdat, all_energy, get_order

# I recomend using command w to close the plot window after each test

@pytest.mark.parametrize("nmax", [5, 10, 15, 100, 500, 1000, 5000])
def test_initdat(nmax):
    arr = initdat(nmax, 1)
    # Check if the array is of the correct shape
    arr = np.array(arr)
    assert arr.shape == (nmax, nmax)
    # Check if the values are within the range [0, 2*pi]
    assert np.all(arr >= 0) and np.all(arr <= 2 * np.pi)

@pytest.mark.parametrize("nmax", [15])
def test_plotdat(nmax):
    arr = np.random.random_sample((nmax, nmax))*2.0*np.pi
    # Test with pflag = 0, 1, 2, 3 to check if it runs without error
    # convert the arr to a a list of lists
    arr = arr.tolist()
    plotdat(arr, 0, nmax)
    plotdat(arr, 1, nmax)
    plotdat(arr, 2, nmax)
    plotdat(arr, 3, nmax)
    
    arr = np.ones((nmax, nmax)) # All should point in the same direction to the top right
    arr = arr.tolist()
    plotdat(arr, 0, nmax)
    plotdat(arr, 1, nmax)
    plotdat(arr, 2, nmax)
    plotdat(arr, 3, nmax)
    # No assertion needed as we are just checking if it runs without error
    
@pytest.mark.parametrize("nmax", [5, 10, 15, 100, 500, 1000, 5000, 10000])
def test_all_energy(nmax):
    # All angles are 0, so energy should be 4 * nmax^2 
    arr = np.ones((nmax, nmax))
    arr = arr.tolist()
    energy = all_energy(arr, nmax, 1)
    # Expecting 4 time the number of elements in the array
    assert energy == -4 * nmax*nmax # Expected energy for a 10x10 array of ones
    
@pytest.mark.parametrize("nmax", [5, 10, 15, 100, 500, 1000, 5000, 10000])
def test_get_order(nmax):
    arr = np.ones((nmax, nmax))
    arr = arr.tolist()
    order = get_order(arr, nmax, 1)
    # Make sure its around 1 (small numerical errors can occur so use round)
    assert round(order,1) == 1.0 # Expected order for a 10x10 array of ones