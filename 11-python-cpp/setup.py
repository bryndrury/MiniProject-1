import os
from setuptools import setup, Extension
from contextlib import contextmanager

# Compile with gcc-14 and g++-14 using the following command:
# CC=gcc-14 CXX=g++-14 python setup.py build_ext --inplace

# -I/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12 -I/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12
# -L/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/config-3.12-darwin -ldl -framework CoreFoundation

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
        sources=["LebwohlLasher.cpp"],
        extra_compile_args=[
            "-fopenmp",
            "-std=c++2a",
            "-O3",
            "-I/opt/homebrew/Cellar/gsl/2.8/include",
            "-I/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12",
            "-I/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12"
            ],
        extra_link_args=[
            "-L/opt/homebrew/Cellar/gsl/2.8/lib",
            "-lgsl",
            "-lgslcblas",
            "-L/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/config-3.12-darwin",
            "-ldl",
            "-framework", "CoreFoundation",
            "-lpython3.12",
            "-L/opt/homebrew/Cellar/libomp/19.1.0/lib",
            "-lomp"],
    )

    setup(
        name="LebwohlLasher",
        version="1.0",
        ext_modules=[module]
    )