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
      scripts=['bin/keplerjc','bin/cbvshrink'],
      packages=['oxksc'],
      ext_modules=[Extension('keplerjc.fmodels', ['src/models.f90'], libraries=['gomp','m'])],
      install_requires=['numpy', 'scipy', 'astropy']
     )
    
