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
import tempfile
import subprocess

from distutils.ccompiler import new_compiler
from distutils.command.build import build as _build
from distutils.errors import CCompilerError, DistutilsOptionError
from distutils.sysconfig import customize_compiler
from setuptools import Extension
from setuptools import setup
from glob import glob


# pyfastnoisesimd version
major_ver = 0
minor_ver = 4
nano_ver = 2

branch = ''

VERSION = "%d.%d.%d%s" % (major_ver, minor_ver, nano_ver, branch)

# Create the version.py file
open('pyfastnoisesimd/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

# Sources and headers
sources = [
    'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD.cpp',
    'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_internal.cpp',
    'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_neon.cpp',
    'pyfastnoisesimd/wrapper.cpp',
]

# For (some versions of) `pip`, the first command run is `python setup.py egg_info` 
# which crashes if `numpy` is not present, so we protect it here.
try:
    from numpy import get_include
    inc_dirs = [get_include(), 'pyfastnoisesimd', 'pyfastnoisesimd/fastnoisesimd/']
except ImportError:
    print('WARNING: NumPy not installed, it is required for compilation.')
    inc_dirs = ['pyfastnoisesimd', 'pyfastnoisesimd/fastnoisesimd/']

lib_dirs = []
libs = []
def_macros = []

with open('README.md') as fh:
    long_desc = fh.read()

with open('requirements.txt') as fh:
    install_requires = [line.strip('\n') for line in fh.readlines()]

if os.name == 'nt':
    extra_cflags = []
    avx512 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_avx512.cpp'
        ],
        'cflags': [
            '/arch:AVX512', '/arch:AVX512F',
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

    if platform.machine() == 'AMD64': # 64-bit windows
        #`/arch:SSE2` doesn't exist on Windows x64 builds, and generates a needless warnings
        sse41 = {
            'sources': [
                'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_sse41.cpp'
            ],
            'cflags': [
            ],
        }
        sse2 = {
            'sources': [
                'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_sse2.cpp'
            ],
            'cflags': [
            ],
        }
    else: # 32-bit Windows
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
    fma_flags = None
else:  # Linux
    extra_cflags = ['-std=c++11']
    avx512 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_avx512.cpp'
        ],
        'cflags': [
            '-std=c++11',
            '-mavx512f',
        ],
    }
    avx2 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_avx2.cpp'
        ],
        'cflags': [
            '-std=c++11',
            '-mavx2',
        ]
    }
    sse41 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_sse41.cpp'
        ],
        'cflags': [
            '-std=c++11',
            '-msse4.1',
        ],
    }
    sse2 = {
        'sources': [
            'pyfastnoisesimd/fastnoisesimd/FastNoiseSIMD_sse2.cpp'
        ],
        'cflags': [
            '-std=c++11',
            '-msse2',
        ],
    }
    fma_flags = ['-mfma']

clibs = [
    ('avx512', avx512),
    ('avx2', avx2),
    ('sse41', sse41),
    ('sse2', sse2),
]


class build(_build):
    user_options = _build.user_options + [
        ('with-avx512=', None, 'Use AVX512 instructions: auto|yes|no'),
        ('with-avx2=', None, 'Use AVX2 instructions: auto|yes|no'),
        ('with-sse41=', None, 'Use SSE4.1 instructions: auto|yes|no'),
        ('with-sse2=', None, 'Use SSE2 instructions: auto|yes|no'),
        ('with-fma=', None, 'Use FMA instructions: auto|yes|no'),
    ]

    def initialize_options(self):
        _build.initialize_options(self)
        self.with_avx512 = 'auto'
        self.with_avx2 = 'auto'
        self.with_sse41 = 'auto'
        self.with_sse2 = 'auto'
        self.with_fma = 'auto'

    def finalize_options(self):
        _build.finalize_options(self)

        compiler = new_compiler(compiler=self.compiler, verbose=self.verbose)
        customize_compiler(compiler)

        disabled_libraries = []

        # Section for custom limits imposed on the SIMD instruction levels based 
        # on the installed compiler
        plat_compiler = platform.python_compiler()
        if plat_compiler.lower().startswith('gcc'):
            # Check the installed gcc version, as versions older than 7.0 claim to
            # support avx512 but are missing some intrinsics that FastNoiseSIMD calls
            output = subprocess.check_output('gcc --version', shell=True)
            gcc_version = tuple([int(x) for x in re.findall( b'\d+(?:\.\d+)+', output)[0].split(b'.')])
            if gcc_version < (7,2): # Disable AVX512
                disabled_libraries.append('avx512')
            if gcc_version < (4,7): # Disable AVX2
                disabled_libraries.append('avx2')
        elif plat_compiler.lower().startswith('msc'):
            # No versions of Windows Python support AVX512 yet
            #                 MSVC++ 14.1 _MSC_VER == 1911 (Visual Studio 2017)
            #                 MSVC++ 14.1 _MSC_VER == 1910 (Visual Studio 2017)
            # Python 3.5/3.6: MSVC++ 14.0 _MSC_VER == 1900 (Visual Studio 2015)
            # Python 3.4:     MSVC++ 10.0 _MSC_VER == 1600 (Visual Studio 2010)
            # Python 2.7:     MSVC++ 9.0  _MSC_VER == 1500 (Visual Studio 2008)
            # Here we just assume the user has the platform compiler
            msc_version = int(re.findall('v\.\d+', plat_compiler)[0].lstrip('v.'))
            # print('FOUND MSVC VERSION: ', msc_version)
            # Still not working with MSVC2017 yet with 1915 and Python 3.7, it 
            # cannot find the function `_mm512_floor_ps`
            if msc_version < 1916:
                disabled_libraries.append('avx512')
            if msc_version < 1900:
                disabled_libraries.append('avx2')
        # End of SIMD limits

        for name, lib in self.distribution.libraries:
            val = getattr(self, 'with_' + name)
            if val not in ('auto', 'yes', 'no'):
                raise DistutilsOptionError('with_%s flag must be auto, yes, '
                                           'or no, not "%s".' % (name, val))

            if val == 'no':
                disabled_libraries.append(name)
                continue

            if not self.compiler_has_flags(compiler, name, lib['cflags']):
                if val == 'yes':
                    # Explicitly required but not available.
                    raise CCompilerError('%s is not supported by your '
                                         'compiler.' % (name, ))
                disabled_libraries.append(name)

        use_fma = False
        if (self.with_fma != 'no' and
                ('avx512' not in disabled_libraries or
                 'avx2' not in disabled_libraries)):
            if fma_flags is None:
                # No flags required.
                use_fma = True
            elif self.compiler_has_flags(compiler, 'fma', fma_flags):
                use_fma = True
                avx512['cflags'] += fma_flags
                avx2['cflags'] += fma_flags
            elif self.with_fma == 'yes':
                # Explicitly required but not available.
                raise CCompilerError('FMA is not supported by your compiler.')

        self.distribution.libraries = [lib
                                       for lib in self.distribution.libraries
                                       if lib[0] not in disabled_libraries]

        with open('pyfastnoisesimd/fastnoisesimd/x86_flags.h', 'wb') as fh:
            fh.write(b'// This file is generated by setup.py, '
                     b'do not edit it by hand\n')
            for name, lib in self.distribution.libraries:
                fh.write(b'#define FN_COMPILE_%b\n' % (name.upper().encode('ascii', )))
            if use_fma:
                fh.write(b'#define FN_USE_FMA\n')

    def compiler_has_flags(self, compiler, name, flags):

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                test_file = 'test-%s.cpp' % (name, )
                with open(test_file, 'w') as fd:
                    fd.write('int main(void) { return 0; }')

                try:
                    compiler.compile([test_file], extra_preargs=flags)
                except CCompilerError:
                    self.warn('Compiler does not support %s flags: %s' %
                              (name, ' '.join(flags)))
                    return False

            finally:
                os.chdir(cwd)

        return True


# List classifiers:
# https://pypi.python.org/pypi?%3Aaction=list_classifiers
classifiers = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Information Technology
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Multimedia :: Graphics :: 3D Modeling
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""

setup(name = "pyfastnoisesimd",
      version = VERSION,
      description = 'Python Fast Noise with SIMD',
      long_description = long_desc,
      long_description_content_type='text/markdown',
      classifiers = [c for c in classifiers.split("\n") if c],
      author = 'Robert A. McLeod',
      author_email = 'robbmcleod@gmail.com',
      maintainer = 'Robert A. McLeod',
      maintainer_email = 'robbmcleod@gmail.com',
      url = 'http://github.com/robbmcleod/pyfastnoisesimd',
      license = 'https://opensource.org/licenses/BSD-3-Clause',
      platforms = ['any'],
      libraries = clibs,
      cmdclass = {'build': build},
      install_requires=install_requires,
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
