from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# setup(ext_modules=cythonize('cpolicy.pyx', language_level=3))
#
# setup(
#     ext_modules=[
#         Extension("cpolicy", ["cpolicy.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )
#
# # Or, if you use cythonize() to make the ext_modules list,
# # include_dirs can be passed to setup()

os.environ["CC"] = "g++"

setup(
    ext_modules=cythonize("*.pyx", language_level=3),
    include_dirs=[numpy.get_include()]
)