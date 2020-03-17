from distutils.core import setup, Extension
import os



setup(name='Mop-cGT',
      version='0.1',
      description='Model-to-observable projection code for galaxy thermodynamics',
      url='https://github.com/samodeo/Mop-c-GT',
      author='Stefania Amodeo',
      author_email='sa649@cornell.edu',
      license='BSD-2-Clause',
      packages=['Mop-cGT'],
      package_dir={'hmvec':'Mop-cGT'},
      zip_safe=False)