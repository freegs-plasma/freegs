from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

import freegs

setup(
    name="FreeGS",
    version=freegs.__version__,
    packages=["freegs"],

    license="LGPL",
    author="Ben Dudson",
    author_email='benjamin.dudson@york.ac.uk',
    url="https://github.com/bendudson/freegs",
    description="Free boundary Grad-Shafranov solver for tokamak plasma equilibria",

    long_description=read("README.md"),
    
    install_requires=['numpy>=1.8',
                      'scipy>=0.14',
                      'matplotlib>=1.3',
                      'h5py>=2.10.0'],
    
    platforms='any',

    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Physics'
        ],
)
