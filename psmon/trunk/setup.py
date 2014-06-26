from setuptools import setup, find_packages

#workaround for LCLS scons compatibility
import os
if not os.path.exists('psmon'):
    os.rename('src','psmon')

setup(
    name='psmon',
    version='0.0.1',
    description='LCLS analysis monitoring',
    author='SLAC RED/PCDS',
    author_email='pcds-ana-l',
    packages=find_packages(),
    #install_requires=['numpy>=1.6.2'],
    entry_points={
        'console_scripts': [
            'psmonserver = psmon.psmonserver:main',
            'psmonclient = psmon.psmonclient:main',
        ]
    },
)
