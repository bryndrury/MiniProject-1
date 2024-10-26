import os
from setuptools import setup, Extension
from contextlib import contextmanager

# Compile with gcc-14 and g++-14 using the following command:
# CC=gcc-14 CXX=g++-14 python3 setup.py build_ext --inplace

##### MAKE SURE TO CHENGE THESE PATHS TO YOUR PYTHON VERSION #####
# ============================================================== #
# Python include and library paths: 
# python3-config --includes
# python3-config --ldflags

@contextmanager
def set_env(**environ):
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
        
with set_env(CC="gcc-14", CXX="g++-14"):
    module = Extension(
        "LebwohlLasher", 
        sources=["LebwohlLasherInterface.cpp"],
                #  "LebwohlLasher.cpp"],
        
        ## UPDATE THESE PATHS FOR YOUR GSL, OPENMP, AND PYTHON ##
        # ===================================================== #
        
        extra_compile_args=[
            "-fopenmp",
            "-std=c++2a",
            "-O3",
            # GSL include path
            "-I/opt/homebrew/Cellar/gsl/2.8/include",
            # Python include path
            "-I/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12",
            "-I/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12"
            ],
        extra_link_args=[
            # GSL library path
            "-L/opt/homebrew/Cellar/gsl/2.8/lib",
            "-lgsl",
            # Python library path
            "-L/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/config-3.12-darwin",
            "-ldl",
            "-framework", "CoreFoundation", # These need to be separated even though they are same 'argument'
            "-lpython3.12",
            # OpenMP library path
            "-L/opt/homebrew/Cellar/libomp/19.1.0/lib",
            "-lomp",
        ]
    )

    setup(
        name="LebwohlLasher",
        version="1.0",
        ext_modules=[module]
    )