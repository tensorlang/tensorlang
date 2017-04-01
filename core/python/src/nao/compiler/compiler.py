import sys

from os import path

import tensorflow as tf

from tensorflow.python.framework import meta_graph

from nao.compiler.asset import compiler as asset_compiler
from nao.compiler.nao import compiler as nao_compiler
from nao.compiler.py import compiler as py_compiler
from nao.compiler.metagraph_pbtxt import compiler as metagraph_pbtxt_compiler

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class Workspace:
  def __init__(self, src_root, pkg_root):
    self._src_root = src_root
    self._pkg_root = pkg_root
    self.clear()

  def clear(self):
    self._source_cache = {}

  def put_src(self, filename, source):
    self._source_cache[filename] = source

  def read_src(self, filename):
    if filename in self._source_cache:
      return self._source_cache[filename]

    filepath = path.join(self._src_root, filename)
    if not path.exists(filepath):
      return None

    with open(filepath) as f:
      return f.read()

  def find_pkg_path(self, filename):
    filepath = path.join(self._pkg_root, filename)
    if not path.exists(filepath):
      return None
    return filepath

class Compiler:
  def __init__(self, src_root, pkg_root):
    self._g = tf.Graph()
    self._workspace = Workspace(src_root, pkg_root)
    self._import_cache = {}
    self._import_cache_tags = {}
    self._compilers = [
      asset_compiler,
      nao_compiler,
      py_compiler,
      metagraph_pbtxt_compiler,
    ]
    self.clear()

  def clear(self):
    self._workspace.clear()

  def put_source(self, filename, source):
    self._workspace.put_src(filename, source)

  def new_session(self):
    return tf.Session(graph=self._g)

  def meta_graph_def(self):
    meta_graph_def = None
    with self._g.as_default():
      for compiler in self._compilers:
        if hasattr(compiler, "finish"):
          compiler.finish()

      meta_graph_def, _ = meta_graph.export_scoped_meta_graph()
    return meta_graph_def

  def resolve_import_path(self, import_path, tags=None, reimport=False):
    pkg = None
    if import_path in self._import_cache:
      pkg = self._import_cache[import_path]
      prev_tags = self._import_cache_tags[import_path]
      if tags is not None:
        if prev_tags is None or sorted(prev_tags.items()) != sorted(tags.items()):
          raise Exception("A subsequent resolution of %s used different tags. Before: %s vs Now %s" % (import_path, prev_tags, tags))
      if not reimport:
        return pkg

    needed_imports, compile_fn = self._resolve_import_path(import_path, tags)

    imports = {}
    for imported_path, imported_tags in needed_imports:
      imports[imported_path] = self.resolve_import_path(imported_path, imported_tags)

    with self._g.as_default():
      pkg = compile_fn(imports, pkg)

    self._import_cache[import_path] = pkg
    self._import_cache_tags[import_path] = tags

    return pkg

  def _resolve_import_path(self, import_path, tags):
    for compiler in self._compilers:
      resolved = compiler.make_compile_fn(self._workspace, import_path, tags or {})
      if resolved:
        return resolved

    raise Exception("No such import path: " + import_path)
