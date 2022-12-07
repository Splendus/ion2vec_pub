from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

sourcefiles = ['word2vec_inner.pyx', 'voidptr.c']

extensions = [Extension('word2vec_inner', sourcefiles, 
    include_dirs=[numpy.get_include()])]

setup(
    ext_modules = cythonize(extensions, include_path=[numpy.get_include()])
)
