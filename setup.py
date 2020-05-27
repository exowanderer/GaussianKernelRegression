try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages, Extension

from numpy import get_include

setup(name='gaussian_kernel_regression',
      version=0.3,
      description='Compute the GKR weights over a 3D time series',
      long_description=open('README.md').read(),
      url='https://github.com/exowanderer/GaussianKernelRegression',
      license='GPL3',
      author="Jonathan Fraine (exowanderer)",
      packages=find_packages(),
      install_requires=['numpy'],
      extras_require={'plots':  ['matplotlib']}
      )
