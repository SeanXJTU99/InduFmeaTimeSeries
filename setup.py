"""Build C extensions for hardware-proximate signal processing.

Extensions:
    _covariance  — lower-triangular covariance pack/unpack (memcpy)
    _compressor  — greedy agglomerative dictionary quantisation

Build:  python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import numpy as np

covariance_ext = Extension(
    "_covariance",
    sources=["src/deploy/_covariance.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native"],
)

compressor_ext = Extension(
    "_compressor",
    sources=["src/signal/_compressor.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native"],
)

setup(
    name="industrial_fmea_agent",
    ext_modules=[covariance_ext, compressor_ext],
)
