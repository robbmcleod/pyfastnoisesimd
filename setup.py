# -*- coding: utf-8 -*-
########################################################################
#
#       pyfastnoisesimd
#       License: BSD
#       Created: August 13, 2017
#       C++ Library Author: Jordan Peck - https://github.com/Auburns
#       Python Extension Author:  Robert A. McLeod - robbmcleod@gmail.com
#
########################################################################

# flake8: noqa
from __future__ import print_function

import os
import platform
import re
import sys

from setuptools import Extension
from setuptools import setup
from glob import glob
from numpy import get_include

# pyfastnoisesimd version
major_ver = 0
minor_ver = 2
nano_ver = 2

branch = ''

VERSION = "%d.%d.%d%s" % (major_ver, minor_ver, nano_ver, branch)

# Create the version.py file
open('pyfastnoisesimd/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

# Sources and headers
sources = [
    'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD.cpp',
    'pyfastnoisesimd/wrapper.cpp'
]
inc_dirs = [get_include(), 'pyfastnoisesimd', 'pyfastnoisesimd/fastnoisesimd/']
lib_dirs = []
libs = []
def_macros = []

with open('README.rst') as fh:
    long_desc = fh.read()

if os.name == 'nt':
    extra_cflags = []
    avx512 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_avx512.cpp'
        ],
        'cflags': [
            '/arch:AVX512',
        ],
    }
    avx2 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_avx2.cpp'
        ],
        'cflags': [
            '/arch:AVX2',
        ]
    }
    sse41 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_sse41.cpp'
        ],
        'cflags': [
            '/arch:SSE2',
        ],
    }
    sse2 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_sse2.cpp'
        ],
        'cflags': [
            '/arch:SSE2',
        ],
    }
else:  # Linux
    extra_cflags = ['-std=c++11']
    avx512 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_avx512.cpp'
        ],
        'cflags': [
            '-mavx512f',
        ],
    }
    avx2 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_avx2.cpp'
        ],
        'cflags': [
            '-mavx2',
            '-mfma',
        ]
    }
    sse41 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_sse41.cpp'
        ],
        'cflags': [
            '-msse4.1',
        ],
    }
    sse2 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_sse2.cpp'
        ],
        'cflags': [
            '-msse2'
        ],
    }

clibs = [
    ('avx512', avx512),
    ('avx2', avx2),
    ('sse41', sse41),
    ('sse2', sse2),
]

# List classifiers:
# https://pypi.python.org/pypi?%3Aaction=list_classifiers
classifiers = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Information Technology
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Multimedia :: Graphics :: 3D Modeling
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""

setup(name = "pyfastnoisesimd",
      version = VERSION,
      description = 'Python Fast Noise with SIMD',
      long_description = long_desc,
      classifiers = [c for c in classifiers.split("\n") if c],
      author = 'Robert A. McLeod',
      author_email = 'robbmcleod@gmail.com',
      maintainer = 'Robert A. McLeod',
      maintainer_email = 'robbmcleod@gmail.com',
      url = 'http://github.com/robbmcleod/pyfastnoisesimd',
      license = 'https://opensource.org/licenses/BSD-3-Clause',
      platforms = ['any'],
      libraries = clibs,
      ext_modules = [
        Extension( "pyfastnoisesimd.extension",
                   include_dirs=inc_dirs,
                   define_macros=def_macros,
                   sources=sources,
                   library_dirs=lib_dirs,
                   libraries=libs,
                   extra_compile_args=extra_cflags),
        ],
      # tests_require=tests_require,
      packages = ['pyfastnoisesimd'],

)
