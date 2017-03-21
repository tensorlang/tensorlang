#!/bin/bash

set -ex
export SOURCE_DATE_EPOCH=315561601

cd $(dirname $0)

if [ -e build ]; then
  chmod +w -R build
fi
rm -rf build dist
mkdir -p build dist

PIP_CACHE=$PWD/.pex-build

# Set up virtualenv to work within
VENV=$PWD/build/venv
virtualenv --never-download $VENV

# Remove spurious rpath from NIX_LDFLAGS.
export NIX_LDFLAGS=$(echo $NIX_LDFLAGS | sed -E 's/-rpath [^ ]+//')

# Install cx_Freeze
$VENV/bin/pip3 install \
    --no-cache-dir \
    --no-index \
    ./tools/cx_Freeze-5.0.1

# Install nao
# Generate JavaScript parser file
mkdir -p gen/nao_parser
env OUTDIR=gen/nao_parser yarn run build-parser
touch gen/nao_parser/__init__.py

$VENV/bin/pip3 install \
    --find-links $PIP_CACHE \
    --no-cache-dir \
    --no-index \
    .

# HACK(adamb) Hard code the proper extension name for py_mini_racer
PY_MINI_RACER_DIR=$VENV/lib/python3.5/site-packages/py_mini_racer
PY_MINI_RACER_EXT=$(basename $(ls $PY_MINI_RACER_DIR/_v8.*.so | head -n 1))
PY_MINI_RACER_PY=$PY_MINI_RACER_DIR/py_mini_racer.py
sed -i.bak -e "s!EXTENSION_NAME = .*!EXTENSION_NAME = '$PY_MINI_RACER_EXT'!" $PY_MINI_RACER_PY
rm $PY_MINI_RACER_PY.bak

# HACK(adamb) Fix missing module message for google.protobuf
touch $VENV/lib/python3.5/site-packages/google/__init__.py

cat <<EOF > $VENV/lib/python3.5/site-packages/tensorflow/contrib/util/loader.py
"""Utilities for loading op libraries.

@@load_op_library
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


def load_op_library(path):
  """Loads a contrib op library from the given path.

  NOTE(adamb): Assume that contrib op libraries are statically linked into the main TensorFlow Python

  Args:
    path: An absolute path to a shared object file.

  Returns:
    A Python module containing the Python wrappers for Ops defined in the
    plugin.
  """
  return None
EOF

$VENV/bin/python ./setup_cx_freeze.py build

OLD_LIBSYSTEM=/nix/store/jbilg3n3iwwggf0ca7zfjgbhgwmzwc2s-Libsystem-osx-10.11.6/lib/libSystem.B.dylib
OLD_CORE_FOUNDATION=/nix/store/6pqkka39jnvc72yifpsbpdm6wqymfwqf-CF-osx-10.9.5/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
OLD_CORE_FOUNDATION2=/nix/store/d86m2phvszp7mc55722liw6zzlzb08fr-CF-osx-10.9.5/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
OLD_SYSTEM_CONFIGURATION=/nix/store/ibx5gmwjnk2kxwgz3mvyvx2f953z5vgw-configd-osx-10.8.5/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration
OLD_LIBNCURSES=/nix/store/7igwzx5prz7yvqmmj0l235wyfcrvf7fi-ncurses-5.9/lib/libncursesw.5.dylib

find build/exe.macosx-*/ \
    \( -iname '*.so' -or -iname '*.dylib' \) \
    -exec \
    chmod u+w '{}' '+'

find build/exe.macosx-*/ \
    \( -iname '*.so' -or -iname '*.dylib' -or -path 'build/exe.macosx-*/bin/nao' \) \
    -exec \
        install_name_tool \
        -change $OLD_LIBSYSTEM /usr/lib/libSystem.B.dylib \
        -change $OLD_SYSTEM_CONFIGURATION /System/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration \
        -change $OLD_CORE_FOUNDATION /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation \
        -change $OLD_CORE_FOUNDATION2 /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation \
        -change $OLD_LIBNCURSES "@rpath/libncursesw.5.dylib" \
        '{}' ';'
