from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

setup(name='CBVshrink',
      version='0.5',
      description='Systematics correction for Kepler light curves using PDC-MAP CBVs with Variational Bayes and shrinkage priors',
      author='Suzanne Aigrain',
      author_email='suzanne.aigrain@gmail.com',
      url='https://github.com/saigrain/CBVshrink',
      package_dir={'cbvshrink':'src'},
      scripts=['bin/cbvshrink'],
      packages=['cbvshrink'],
      install_requires=['numpy', 'scipy', 'astropy']
     )

