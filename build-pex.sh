#!/bin/bash

set -ex
export SOURCE_DATE_EPOCH=315532800

cd $(dirname $0)

rm -r build dist
mkdir -p gen/nao_parser build dist

# Generate JavaScript parser file
env OUTDIR=gen/nao_parser yarn run build-parser
touch gen/nao_parser/__init__.py

mkdir -p build/pex-cache

# Open and fix tensorflow .whl
TF_NAME=tensorflow_gpu
TF_VERSION=1.0.0
mkdir -p build/fix-tensorflow-pex
cd build/fix-tensorflow-pex
TF_WHL=$(ls ../../.pex-build/$TF_NAME-$TF_VERSION-*.whl | head -n 1)
unzip $TF_WHL

# Need to strip "-$TF_VERSION.data"
mv $TF_NAME-$TF_VERSION.data/purelib/* .
rmdir $TF_NAME-$TF_VERSION.data/purelib
rmdir $TF_NAME-$TF_VERSION.data

# Need to fix file names in $TF_NAME-$TF_VERSION.dist-info/RECORD
TENSORFLOW_RECORD=$TF_NAME-${TF_VERSION}.dist-info/RECORD
sed -i.bak -e "s!^$TF_NAME-$TF_VERSION.data/tensorflow/purelib!tensorflow!" $TENSORFLOW_RECORD
rm $TENSORFLOW_RECORD.bak

# Ensure Root-Is-Purelib is true now.
TF_DIST_INFO_WHEEL=$TF_NAME-${TF_VERSION}.dist-info/WHEEL
sed -i.bak -e "s!Root-Is-Purelib: false!Root-Is-Purelib: true!" $TF_DIST_INFO_WHEEL
rm $TF_DIST_INFO_WHEEL.bak
cat $TF_DIST_INFO_WHEEL

# Fix rpath for TensorFlow native library.
TF_NATIVE_LIB=tensorflow/python/_pywrap_tensorflow.so
CUDA_HOME=/usr/local/cuda
install_name_tool -add_rpath $CUDA_HOME/lib $TF_NATIVE_LIB
install_name_tool -add_rpath $CUDA_HOME/extras/CUPTI/lib $TF_NATIVE_LIB

zip -r ../pex-cache/$(basename $TF_WHL) .

cd -

# Symlink in any pex files in cache that haven't been patched.
cd build/pex-cache
for dep in $(ls ../../.pex-build/*); do
  if [ ! -e $(basename $dep) ]; then
    ln -s $dep .
  fi
done
cd -

# Generate .pex
pex \
    -vvvvv \
    --no-index \
    --script nao \
    --cache-dir build/pex-cache/ \
    -r requirements-transitive.txt \
    -o dist/nao.pex \
    .
