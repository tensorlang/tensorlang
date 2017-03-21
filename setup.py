from glob import glob
import os
from setuptools import setup, find_packages

_version = os.getenv('BUILD_COUNTER', '9')

def read(filename):
  with open(os.path.join(os.path.dirname(__file__), filename)) as f:
    return f.read()

setup(
    name='nao',
    author='Adam Bouhenguel',
    author_email='adam@bouhenguel.com',
    url='https://github.com/ajbouh/nao',
    version=_version,
    install_requires=[l for l in read('requirements.txt').splitlines() if l and not l.startswith("#")],
    packages=[*find_packages('src'), *find_packages('gen')],
    package_dir={'nao_parser': 'gen/nao_parser', '': 'src'},
    entry_points={
      "console_scripts": [
        "nao = nao.cli:main"
      ]
    },
    include_package_data=True,
    # py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
)
