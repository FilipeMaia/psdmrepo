from setuptools import setup, find_packages
from distutils.core import setup, Extension
from distutils.sysconfig import get_config_var
import numpy

pypsalg_c = Extension('pypsalg_c',
                    library_dirs = [get_config_var('LIBDIR')],
                    include_dirs = [numpy.get_include(),"pyext"],
                    sources = ["pyext/pypsalg_c.c",
                               "pyext/chi2.c", 
                               "pyext/ScaleByN.c"])

#workaround for LCLS scons compatibility
import os
if not os.path.exists('pypsalg'):
    os.rename('src','pypsalg')

setup(name='LCLS',
      version='0.0',
      description='LCLS analysis algorithms and objects',
      author='SLAC RED/PCDS',
      author_email='pcds-ana-l',
      packages=find_packages(),
#      install_requires=['numpy>=1.6.2'],
      ext_modules = [pypsalg_c]
)
