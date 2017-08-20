# -*- coding: utf-8 -*-
########################################################################
#
#       PyFastNoiseSIMD
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

sys.path.append( os.path.join(os.path.abspath('.'), 'PyFastNoiseSIMD')  )
import cpuinfo # Is up one directory

from setuptools import Extension
from setuptools import setup
from glob import glob
from numpy import get_include


# FastNoiseSIMD version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('PyFastNoiseSIMD/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()

# Sources and headers
sources = [ 'PyFastNoiseSIMD/FastNoiseSIMD/FastNoiseSIMD.cpp',
            'PyFastNoiseSIMD/FastNoiseSIMD/FastNoiseSIMD_internal.cpp',
            'PyFastNoiseSIMD/wrapper.cpp' ]
inc_dirs = [get_include()]
lib_dirs = []
libs = []
def_macros = []

# Auto-detect the architecture and set the CFLAGS appropriately.
cpu_info = cpuinfo.get_cpu_info()
FOUND_SSE2 = False
FOUND_AVX2 = False
# FMA is only enabled if AVX2 is also enabled
FOUND_FMA = False
FOUND_AVX512 = False
FOUND_NEON = False
print( '''--==NOTE: AVX512 and FMA cannot be AUTO-DETECTED at present.  Modify setup.py
vars `FOUND_AVX512` and `FOUND_FMA` to force compilation. ==--''' )
if 'sse2' in cpu_info['flags']:
    FOUND_SSE2 = True
if 'avx2' in cpu_info['flags']:
    FOUND_AVX2 = True
# TODO: Need a new cpuinfo.py to detect AVX512 and FMA support.

sse_arch = b'''\
#define FN_COMPILE_SSE2
#define FN_COMPILE_SSE41

'''
avx2_arch = b'''\
// To compile AVX2 set C++ code generation to use /arch:AVX(2) on FastNoiseSIMD_avx2.cpp
// Note: This does not break support for pre AVX CPUs, AVX code is only run if support is detected
#define FN_COMPILE_AVX2

'''
avx512_arch = b'''\
// Only the latest compilers will support this
#define FN_COMPILE_AVX512

'''
fma_arch = b'''\
// Using FMA instructions with AVX(51)2/NEON provides a small performance increase but can cause 
// minute variations in noise output compared to other SIMD levels due to higher calculation precision
// Intel compiler will always generate FMA instructions, use /Qfma- or -no-fma to disable
#define FN_USE_FMA

'''

with open( 'PyFastNoiseSIMD/FastNoiseSIMD/amd64_arch.h', 'wb') as fh:

    if os.name == 'nt':
        if FOUND_SSE2:
            sources += ['PyFastNoiseSIMD/FastNoiseSIMD/FastNoiseSIMD_sse2.cpp', 'PyFastNoiseSIMD/FastNoiseSIMD/FastNoiseSIMD_sse41.cpp' ]
            # The /arch:SSE2 flag is unnecessary on AMD64 architecture and emits a useless warning.
            # CFLAGS += ['/arch:SSE2']
            fh.write( sse_arch )
        if FOUND_AVX2:
            sources += ['PyFastNoiseSIMD/FastNoiseSIMD/FastNoiseSIMD_avx2.cpp']
            CFLAGS += ['/arch:AVX2']
            fh.write( avx2_arch )
            if FOUND_FMA:
                #  Like AVX512 it's not well supported on Win.
                print( "TODO: detect FMA support on Windows ")
                fh.write( fma_arch )
        if FOUND_AVX512:
            # Technically no Windows Python compiler supports AVX512, because 
            # only MSVC2017 supports it and Python 3.6 runs on MSVC2015.
            print( "TODO: detect AVX512 support on Windows" )
            fh.write( avx512_arch )

        if FOUND_NEON: # Detected by compiler flags
            pass
    else: # Linux
        CFLAGS += ['-std=c++11']
        if FOUND_SSE2:
            sources += ['PyFastNoiseSIMD/FastNoiseSIMD/FastNoiseSIMD_sse2.cpp', 'PyFastNoiseSIMD/FastNoiseSIMD/FastNoiseSIMD_sse41.cpp' ]
            CFLAGS += ['-msse2', '-msse4.1']
            fh.write( sse_arch )
        if FOUND_AVX2:
            sources += ['PyFastNoiseSIMD/FastNoiseSIMD/FastNoiseSIMD_avx2.cpp']
            CFLAGS += ['-mavx2']
            fh.write( avx2_arch )
            if FOUND_FMA:
                print( "TODO: detect FMA support on Linux ")
                CFLAGS += ['-mfma']
                fh.write( fma_arch )
        if FOUND_AVX512:
            print( "TODO: detect AVX512 support on Linux" )
            # Not sure if anything but the foundational functions are called...
            CFLAGS += ['-mavx512f']
            fh.write( avx512_arch )

        if FOUND_NEON: # Detected by compiler flags
            pass

classifiers = """\
Development Status :: 4 - Beta 
Intended Audience :: Developers
Intended Audience :: Information Technology
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Simulation :: Noise
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""

setup(name = "pyfastnoisesimd",
      version = VERSION,
      description = 'FastNoiseSIMD',
      long_description = """\

FastNoise SIMD is the SIMD implementation of my noise library FastNoise. It aims
to provide faster performance through the use of intrinsic(SIMD) CPU functions. 
Vectorisation of the code allows noise functions to process data in sets of 
4/8/16 increasing performance by 700% in some cases (Simplex).

""",
      classifiers = [c for c in classifiers.split("\n") if c],
      author = 'Robert A. McLeod',
      author_email = 'robbmcleod@gmail.com',
      maintainer = 'Robert A. McLeod',
      maintainer_email = 'robbmcleod@gmail.com',
      url = 'http://github.com/robbmcleod/pyfastnoisesimd',
      license = 'https://opensource.org/licenses/BSD-3-Clause',
      platforms = ['any'],
      ext_modules = [
        Extension( "pyfastnoisesimd.extension",
                   include_dirs=inc_dirs,
                   define_macros=def_macros,
                   sources=sources,
                   library_dirs=lib_dirs,
                   libraries=libs,
                   extra_link_args=LFLAGS,
                   extra_compile_args=CFLAGS ),
        ],
      # tests_require=tests_require,
      packages = ['pyfastnoisesimd'],

)
