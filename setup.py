#from setuptools import find_packages
from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

setup(name='OxKeplerSC',
      version='0.9',
      description='Systematics correction for Kepler light curves using PDC-MAP CBVs with Variational Bayes and shrinkage priors',
      author='Suzanne Aigrain',
      author_email='suzanne.aigrain@gmail.com',
      url='https://github.com/OxES/OxKeplerSC',
      package_dir={'oxksc':'src'},
      scripts=['bin/keplerjc','bin/keplersc'],
      packages=['oxksc','oxksc.cbvc','oxksc.jc'],
      ext_modules=[Extension('oxksc.jc.fmodels', ['src/jc/models.f90'], libraries=['gomp','m'])],
      install_requires=['numpy', 'scipy', 'astropy']
     )
    
