import pprint
import sys
import pkgutil

import tensorflow as tf

from py_mini_racer import py_mini_racer

from nao.compiler.nao import graph_gen

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

pp = pprint.PrettyPrinter(indent=2, stream=sys.stderr).pprint

_js_ctx = py_mini_racer.MiniRacer()
_js_ctx.eval(pkgutil.get_data("nao_parser", "parse.js"))

def _parse_import_tag(import_tag):
  if not import_tag:
    return None

  def split_tag(s):
    result = s.split(":", 1)
    if len(result) == 1:
      result = [result[0], True]
    return result

  return dict([split_tag(frag) for frag in import_tag.split(",")])

def _enumerate_imports(exprs):
  imported = []
  for expr in exprs:
    if expr[0] != "_sf_import":
      continue

    for import_name, import_path, import_tag in expr[1]:
      imported.append((import_path, _parse_import_tag(import_tag)))

  return imported

def make_compile_fn(workspace, import_path, tags):
  source = workspace.read_src(import_path + ".nao")
  if source is None:
    return None

  exprs = _js_ctx.call("parse.parseExpressions", source)
  # pp(exprs)

  imported = []
  for imported_path, imported_tags in _enumerate_imports(exprs):
    # Skip imports that provide direct access to TensorFlow internals.
    if imported_path.startswith("tensorflow:"):
      continue

    if imported_tags and imported_tags.get("asset", False):
      imported_tags["url"] = imported_path
      imported_path = imported_path.split("://", 1)[1]

    imported.append((imported_path, imported_tags))

  def compile(resolved_imports, previous):
    return graph_gen.TopLevel()._sf_package(resolved_imports, previous, import_path, *exprs)

  return (imported, compile)
