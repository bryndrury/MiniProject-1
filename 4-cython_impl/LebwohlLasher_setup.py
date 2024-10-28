from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# python LebwohlLasher_setup.py build_ext --inplace

ext_modules = [
    Extension(
        "LebwohlLasher_c",
        ["LebwohlLasher_c.pyx"],
        include_dirs=[
            np.get_include(),
            '/opt/homebrew/opt/libomp/include',
            '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include'
        ],
        extra_compile_args=[
            '-Xpreprocessor', '-fopenmp',
            '-I/opt/homebrew/opt/libomp/include',
            '-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include',
            '-O3'
        ],
        extra_link_args=[
            '-L/opt/homebrew/opt/libomp/lib',
            '-lomp',
            '-Wl,-rpath,/opt/homebrew/opt/libomp/lib'
        ],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
]

setup(
    name="LebwohlLasher_c",
    ext_modules=cythonize(ext_modules),
)

