import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

# python LebwohlLasher_setup.py build_ext --inplace

os.environ["CC"] = "g++-14"
os.environ["CXX"] = "g++-14"

omp_include = os.popen('brew --prefix libomp').read().strip() + '/include'
omp_lib = os.popen('brew --prefix libomp').read().strip() + '/lib'

extensions = [
    Extension(
        "LebwohlLasher_cython",
        ["LebwohlLasher.pyx"],
        include_dirs=[omp_include, np.get_include()],
        extra_compile_args=["-fopenmp", "-std=c++2a", "-O3"],
        extra_link_args=[f'-L{omp_lib}', '-lomp', '-fopenmp'],
    )
]

setup(
    name="LebwohlLasher_cython",
    ext_modules=cythonize(extensions),
    version="0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
    ],
)