#!/bin/bash

# SDIST
python setup.py sdist bdist_wheel upload -r pypi
# Apparently one should use twine now:
# https://packaging.python.org/tutorials/distributing-packages/#packaging-your-project