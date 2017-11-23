from setuptools import setup

import freegs

setup(
    name="FreeGS",
    version=freegs.__version__,
    packages=["freegs"],

    license="LGPL",
    author="Ben Dudson",
    author_email='benjamin.dudson@york.ac.uk',
    url="https://github.com/bendudson/freegs",
    description="Free boundary Grad-Shafranov solver",
    
    install_requires=['numpy>=1.8',
                      'scipy>=0.14',
                      'matplotlib>=1.3'],
    
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
