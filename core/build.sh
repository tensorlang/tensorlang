#!/bin/bash

set -ex
export SOURCE_DATE_EPOCH=315561601

cd $(dirname $0)

VENV=$PWD/build/venv

# Remove spurious rpath from NIX_LDFLAGS.
export NIX_LDFLAGS=$(echo $NIX_LDFLAGS | sed -E 's/-rpath [^ ]+//')

# Check syntax of python code.
find $PWD/python/src -type f -iname '*.py' -exec python -m py_compile {} '+'

if [ -e build ]; then
  chmod +w -R build
fi

rm -rf build dist
mkdir -p build dist

# Set up virtualenv to work within
virtualenv --never-download $VENV

# Install cx_Freeze
$VENV/bin/pip3 install \
    --no-cache-dir \
    --no-index \
    $PWD/native/vendor/cx_Freeze-5.0.1

# Generate JavaScript parser file
export GEN_NAO_PARSER=$PWD/python/gen/nao_parser/parse.js
mkdir -p $(dirname $GEN_NAO_PARSER)
cd $PWD/javascript
yarn run build-parser
cd -
touch $(dirname $GEN_NAO_PARSER)/__init__.py

# Install nao
PIP_CACHE=$PWD/python/vendor/cache

pip_install() {
  pip_venv=$1
  shift
  $pip_venv/bin/pip3 install \
      --no-cache-dir \
      --find-links $PIP_CACHE \
      --no-index \
      --upgrade \
      "$@"
}

pip_install $VENV $PWD/python

PYTHON_NS=python3.5
# HACK(adamb) Hard code the proper extension name for py_mini_racer
PY_MINI_RACER_DIR=$VENV/lib/$PYTHON_NS/site-packages/py_mini_racer
PY_MINI_RACER_EXT=$(basename $(ls $PY_MINI_RACER_DIR/_v8.*.so | head -n 1))
PY_MINI_RACER_PY=$PY_MINI_RACER_DIR/py_mini_racer.py
sed -i.bak -e "s!EXTENSION_NAME = .*!EXTENSION_NAME = '$PY_MINI_RACER_EXT'!" $PY_MINI_RACER_PY
rm $PY_MINI_RACER_PY.bak

# HACK(adamb) Fix missing module message for google.protobuf
touch $VENV/lib/$PYTHON_NS/site-packages/google/__init__.py

cat <<EOF > $VENV/lib/$PYTHON_NS/site-packages/tensorflow/contrib/util/loader.py
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

  Args:
    path: An absolute path to a shared object file.

  Returns:
    A Python module containing the Python wrappers for Ops defined in the
    plugin.
  """
  path = path.replace("//site-packages", os.environ["TF_LIBRARY_DIR"])

  path = resource_loader.get_path_to_datafile(path)
  ret = load_library.load_op_library(path)
  assert ret, 'Could not load %s' % path
  return ret
EOF

TARGET_DIR=build/exe.macosx-10.6-x86_64-3.5

if [ -e $TARGET_DIR ]; then
  chmod -R +w $TARGET_DIR
  rm -rf $TARGET_DIR
fi
mkdir $TARGET_DIR

# Compute differing files between CPU and GPU versions of TensorFlow.
# Copy GPU-specific files to proper secondary location.

rm -rf build/tf-cpu
virtualenv --never-download build/tf-cpu
pip_install build/tf-cpu tensorflow==1.1.0rc2

rm -rf build/tf-gpu
virtualenv --never-download build/tf-gpu
pip_install build/tf-gpu tensorflow_gpu==1.1.0rc2

TF_COMMON_SUBPATH=lib/$PYTHON_NS/site-packages/tensorflow
TF_EXISTING_DIR=$VENV/$TF_COMMON_SUBPATH
TF_GPU_OVERLAY=$(cd $TARGET_DIR; pwd)/lib/$PYTHON_NS-gpu/tensorflow
TF_CPU_OVERLAY=$(cd $TARGET_DIR; pwd)/lib/$PYTHON_NS-cpu/tensorflow
rm -rf $TF_GPU_OVERLAY $TF_CPU_OVERLAY
mkdir -p $TF_GPU_OVERLAY $TF_CPU_OVERLAY
for f in $(diff --brief --new-file -r -x '*.pyc' -x RECORD -x '*.dist-info' build/tf-cpu/$TF_COMMON_SUBPATH build/tf-gpu/$TF_COMMON_SUBPATH | cut -d' ' -f2 | cut -d/ -f7-); do
  if [ -e $TF_EXISTING_DIR/$f ]; then
    rm $TF_EXISTING_DIR/$f
  fi

  if [ -e build/tf-gpu/$TF_COMMON_SUBPATH/$f ]; then
    mkdir -p $TF_GPU_OVERLAY/$(dirname $f)
    cp build/tf-gpu/$TF_COMMON_SUBPATH/$f $TF_GPU_OVERLAY/$f
  fi
  if [ -e build/tf-cpu/$TF_COMMON_SUBPATH/$f ]; then
    mkdir -p $TF_CPU_OVERLAY/$(dirname $f)
    cp build/tf-cpu/$TF_COMMON_SUBPATH/$f $TF_CPU_OVERLAY/$f
  fi
done

referenced_libs() {
  otool -L $1 | tail -n +2 | cut -d' '  -f1
}

NIX_LIBSYSTEM_OSX_DIR=$(nix-store -qR $(which python) | grep Libsystem-osx- | head -n 1)
NIX_LIBSYSTEM_OSX_DIR2=$(dirname $(dirname $(referenced_libs $VENV/lib/$PYTHON_NS/site-packages/tornado/speedups.cpython-35m-darwin.so | grep Libsystem-osx- | head -n 1)))
NIX_CF_OSX_DIR=$(nix-store -qR $(which python) | grep CF-osx- | head -n 1)
NIX_CF_OSX_DIR2=$(nix-store -qR $(which python) | grep CF-osx- | tail -n +2)
NIX_CONFIGD_OSX_DIR=$(nix-store -qR $(which python) | grep configd-osx- | head -n 1)

env \
    CXFREEZE_BIN_PATH_EXCLUDES="$NIX_LIBSYSTEM_OSX_DIR/:$NIX_LIBSYSTEM_OSX_DIR2/:$NIX_CF_OSX_DIR/:$NIX_CF_OSX_DIR2/:$NIX_CONFIGD_OSX_DIR/" \
    $VENV/bin/python ./native/setup.py build

OLD_LIBSYSTEM=$NIX_LIBSYSTEM_OSX_DIR/lib/libSystem.B.dylib
OLD_LIBSYSTEM2=NIX_LIBSYSTEM_OSX_DIR2/lib/libSystem.B.dylib
OLD_CORE_FOUNDATION=$NIX_CF_OSX_DIR/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
OLD_CORE_FOUNDATION2=$NIX_CF_OSX_DIR2/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
OLD_SYSTEM_CONFIGURATION=$NIX_CONFIGD_OSX_DIR/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration
OLD_LIBNCURSES=$(ncurses-config --libdir)/libncursesw.$(ncurses-config --abi-version).dylib

find $TARGET_DIR/ \
    \( -iname '*.so' -or -iname '*.dylib' \) \
    -exec \
    chmod u+w '{}' '+'

find $TARGET_DIR/ \
    \( -iname '*.so' -or -iname '*.dylib' -or -path "$TARGET_DIR/bin/nao" \) \
    -exec \
        install_name_tool \
        -change $OLD_LIBSYSTEM /usr/lib/libSystem.B.dylib \
        -change $OLD_LIBSYSTEM2 /usr/lib/libSystem.B.dylib \
        -change $OLD_SYSTEM_CONFIGURATION /System/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration \
        -change $OLD_CORE_FOUNDATION /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation \
        -change $OLD_CORE_FOUNDATION2 /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation \
        -change $OLD_LIBNCURSES "@rpath/libncursesw.5.dylib" \
        '{}' ';'

# Ensure that we look in the proper place for CUDA dependencies.
install_name_tool \
    -add_rpath /usr/local/cuda/lib \
    $TARGET_DIR/bin/nao
