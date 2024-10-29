import numpy as np
import matplotlib.pyplot as plt
from LebwohlLasher import initdat, plotdat, all_energy, get_order

def test_initdat():
    nmax = 10
    arr = initdat(nmax)
    # Check if the array is of the correct shape
    assert arr.shape == (nmax, nmax)
    # Check if the values are within the range [0, 2*pi]
    assert np.all(arr >= 0) and np.all(arr <= 2 * np.pi)

def test_plotdat_no_plot():
    nmax = 10
    arr = initdat(nmax)
    # Test with pflag = 0 (no plot)
    plotdat(arr, 0, nmax)
    # No assertion needed as we are just checking if it runs without error

def test_plotdat_energy_plot():
    nmax = 10
    arr = initdat(nmax)
    # Test with pflag = 1 (energy plot)
    plotdat(arr, 1, nmax)
    # No assertion needed as we are just checking if it runs without error

def test_plotdat_angles_plot():
    nmax = 10
    arr = initdat(nmax)
    # Test with pflag = 2 (angles plot)
    plotdat(arr, 2, nmax)
    # No assertion needed as we are just checking if it runs without error

def test_plotdat_black_plot():
    nmax = 10
    arr = initdat(nmax)
    # Test with pflag = 3 (black plot)
    plotdat(arr, 3, nmax)
    # No assertion needed as we are just checking if it runs without error
    
def test_all_energy():
    nmax = 10
    
    arr = np.ones((nmax, nmax))
    
    energy = all_energy(arr, nmax)
    
    assert energy == -400 # Expected energy for a 10x10 array of ones
    
def test_get_order():
    nmax = 10
    arr = np.ones((nmax, nmax))
    
    order = get_order(arr, nmax)
    
    assert order == 1.0000000000000002 # Expected order for a 10x10 array of ones