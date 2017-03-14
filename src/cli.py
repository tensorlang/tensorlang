from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import argparse
import json
import pprint
import re
import sys
import traceback
import gc

from os import path

import graph_gen
import graph_io
import graph_query
import graph_xform
import graph_repl
import graph_execution

import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import control_flow_pb2

import subprocess

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def compile_meta_graph(input_json):
  with open(input_json, "r") as f:
    s = f.read()
    input_exprs = json.loads(s)
    # pp.pprint(input_exprs)

    return graph_gen.meta_graph_def_from_exprs(input_exprs)

from py_mini_racer import py_mini_racer

class PalletParser:
  def __init__(self, src_root, pkg_root):
    self._pallet = []
    self._source_cache = {}
    self._import_cache = {}

    self._attempts = [
      {
        "language": "nao",
        "suffix": ".nao",
        "dir": src_root,
      },
      {
        "language": "python",
        "suffix": ".py",
        "dir": src_root,
      },
      {
        "language": "tensorflow:metagraph:pbtxt",
        "suffix": ".metagraph.pbtxt",
        "dir": pkg_root,
      },
    ]

  def pallet(self):
    return self._pallet[:]

  def put_source(self, import_path, source):
    self._source_cache[import_path] = source

  def _enumerate_imports(self, package_expr):
    if package_expr[0] != "_sf_package":
      raise Exception("Not a package expression: " + package_expr)

    name, *exprs = package_expr[1:]
    imported = []
    for expr in exprs:
      if expr[0] != "_sf_import":
        continue

      for import_name, import_path, imported_scope in expr[1]:
        imported.append((import_path, imported_scope))

    return imported

  # TODO(adamb) Modify to *only* take input via stdin/command-line.
  def _parse_external(self, source):
    ctx = py_mini_racer.MiniRacer()
    with open("lib/parse.js") as f:
      ctx.eval(f.read())
    expr = ctx.call("parse.parseExpressions", source)
    # pp(expr)
    return expr

  def _attempt(self, language, import_path, imported_scope, filepath=None, content=None):
    return {
      "language": language,
      "import_path": import_path,
      "imported_package_name": path.basename(import_path),
      "imported_scope_name": imported_scope,
      "path": filepath,
      "content": content,
    }

  def _resolve_import(self, import_path, imported_scope):
    if import_path in self._source_cache:
      source = self._source_cache[import_path]
      return self._attempt("nao", import_path, imported_scope, content=source)

    for attempt in self._attempts:
      filepath = path.join(attempt["dir"], import_path + attempt["suffix"])
      if path.isfile(filepath):
        return self._attempt(
            attempt["language"],
            import_path,
            imported_scope,
            filepath=filepath)

    raise Exception("No such import path: " + import_path)

  def resolve_import(self, import_path, imported_scope):
    cache_key = (import_path, imported_scope)
    if cache_key in self._import_cache:
      return self._import_cache[cache_key]

    resolved = self._resolve_import(import_path, imported_scope)
    if resolved["language"] == "nao":
      source = resolved["content"]
      if source is None:
        with open(resolved["path"]) as f:
          source = f.read()
      exprs = self._parse_external(source)

      pkg = ["_sf_package", resolved["imported_package_name"], *exprs]
      for import_path, imported_scope in self._enumerate_imports(pkg):
        # Skip imports that provide direct access to TensorFlow internals.
        if import_path == "tensorflow":
          continue

        self.resolve_import(import_path, imported_scope)
    else:
      pkg = ["_sf_foreign_package", resolved["language"], resolved["imported_package_name"], resolved["imported_scope_name"], resolved["path"]]

    self._import_cache[cache_key] = pkg
    self._pallet.append(pkg)
    return pkg

def parse_packages(src_root, pkg_root, package_names):
  p = PalletParser(src_root, pkg_root)

  for package_name in package_names:
    p.resolve_import(package_name, None)

  return p.pallet()

def parse_source(src_root, pkg_root, package_name, source):
  p = PalletParser(src_root, pkg_root)
  p.put_source(package_name, source)
  p.resolve_import(package_name, None)

  return p.pallet()

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("package_names", type=str, nargs='*')
  parser.add_argument("--root", type=str, default=".",
                      help="""Specify root directory to search for imports from""")
  parser.add_argument("--source", type=str,
                      help="""Specify source code instead of reading from file""")

  parser.add_argument("--metagraphdef", nargs='?', type=str,
                      help="""Graph file to load.""")
  parser.add_argument("--binary-metagraphdef", nargs='?', type=bool, default=False,
                      help="""Whether or not input is binary.""")
  parser.add_argument("--feed-constants", nargs='?', type=str,
                      help="""Path to GraphDef protobuf with constants to feed""")
  parser.add_argument("--feed-constants-strip", nargs='?', type=str, default="",
                      help="""Prefix to filter for (and strip from) constants""")
  parser.add_argument("--feed-constants-prefix", nargs='?', type=str,
                      help="""Prefix to add to constant names in feed""")
  parser.add_argument("--feed-constants-binary", nargs='?', type=bool, default=False,
                      help="""Whether or not feed constant protobuf is binary""")

  parser.add_argument("--run", default=False, action='store_const', const=True,
                      help="""Run the graph with given (or default) --result* and --feed-* options""")
  parser.add_argument("--run-result-pattern", nargs='?', type=str, default="^(${package}/Main)/outputs/(.*)$",
                      help="""Pattern to discover run results.""")
  parser.add_argument("--result-binary", default=False, action='store_const', const=True,
                      help="""Whether or not to result in binary.""")
  parser.add_argument("--result", nargs='?', type=str, default="/dev/stdout")

  parser.add_argument("--test", default=False, action='store_const', const=True,
                      help="""Run the tests graphs with given (or default) --test-* options""")
  parser.add_argument("--test-result-pattern", nargs='?', type=str, default="^(${package}/Test[^/]*)/outputs/(.*)$",
                      help="""Pattern to discover test graph results.""")

  parser.add_argument("--repl", default=False, action='store_const', const=True,
                      help="""Start REPL""")

  parser.add_argument("--train", default=False, action='store_const', const=True,
                      help="""Run train graphs with given (or default) --train-* options""")
  parser.add_argument("--train-result-pattern", nargs='?', type=str, default="^(${package}/Train[^/]*)/outputs/(.*)$",
                      help="""Pattern to discover train graph results.""")

  parser.add_argument("--input-json", nargs='?', type=str,
                      help="""JSON file to load.""")

  parser.add_argument("--output", default=False, action='store_const', const=True,
                      help="""Output graph""")
  parser.add_argument("--output-root", type=str, default=".",
                      help="""When automatically constructing output path, use this as base""")
  parser.add_argument("--output-name", type=str,
                      help="""Base name to use for output file name. Defaults to ${package} if there's only one.""")
  parser.add_argument("--output-result-pattern", nargs='?', type=str, default="^(${package}/[^/]*)(/outputs/[^/]*)?$",
                      help="""Pattern to discover outputs of graph to output.""")
  parser.add_argument("--output-format", nargs='?', type=str, default="metagraph",
                      help="""Defaults to metagraph""")
  parser.add_argument("--output-binary", default=False, action='store_const', const=True,
                      help="""Whether or not to output in binary.""")
  parser.add_argument("--output-file", nargs='?', type=str,
                      help="""Path to write output to. Defaults to ${output-name}.${output-format}""")

  FLAGS = parser.parse_args()

  package_names = FLAGS.package_names


  should_parse = len(package_names) > 0 or FLAGS.source
  if not (should_parse or FLAGS.run or FLAGS.test or FLAGS.output):
    if os.isatty(1):
      FLAGS.repl = True

  if should_parse and not (FLAGS.repl or FLAGS.run or FLAGS.test or FLAGS.output):
    FLAGS.output = True

  meta_graph_def = None

  output_package_names = None

  if should_parse:
    if FLAGS.source:
      package_names = ["main"]
      expressions = parse_source(FLAGS.root, FLAGS.output_root, package_names[0], FLAGS.source)
    else:
      # Look for matching packages _train
      if FLAGS.train:
        output_package_names = package_names[:]
        package_names.extend([pkg + "_train" for pkg in package_names])
      expressions = parse_packages(FLAGS.root, FLAGS.output_root, package_names)

    # print("parsed", expressions)
    meta_graph_def = graph_gen.meta_graph_def_from_exprs(expressions)
    # We need to do this so we clean up references to py_funcs. LAME.
    gc.collect()

  # Sometimes we want to output different packages than we're testing, training, etc.
  if output_package_names == None:
    output_package_names = package_names

  if not FLAGS.output_name and len(output_package_names) == 1:
    FLAGS.output_name = output_package_names[0]
    if FLAGS.train:
      FLAGS.output_name += "_trained"

  if FLAGS.metagraphdef:
    package_names = ["[^/]+"]
    meta_graph_def = graph_io.read_meta_graph_def(
        FLAGS.metagraphdef,
        FLAGS.binary_metagraphdef)

  if FLAGS.input_json:
    package_names = ["[^/]+"]
    meta_graph_def = compile_meta_graph(FLAGS.input_json)

  if FLAGS.output and FLAGS.output_name and not FLAGS.output_file:
    output_suffix = "." + FLAGS.output_format + ".pb"
    if not FLAGS.output_binary:
      output_suffix += "txt"
    FLAGS.output_file = FLAGS.output_root + "/" + FLAGS.output_name + output_suffix

  # Now that we know our package names, use them to target the proper results.
  package_pattern = "(?:" + str.join("|", package_names) + ")"
  FLAGS.test_result_pattern = FLAGS.test_result_pattern.replace("${package}", package_pattern)
  FLAGS.train_result_pattern = FLAGS.train_result_pattern.replace("${package}", package_pattern)
  FLAGS.run_result_pattern = FLAGS.run_result_pattern.replace("${package}", package_pattern)

  output_package_pattern = "(?:" + str.join("|", output_package_names) + ")"
  FLAGS.output_result_pattern = FLAGS.output_result_pattern.replace("${package}", output_package_pattern)

  if FLAGS.train:
    def post_train(session, result_scope_prefixes):
      graph = session.graph

      trained_var_name_bs = set()
      for result_scope_prefix in result_scope_prefixes:
        collection_name = "%s:variable_names" % result_scope_prefix
        eprint("collection_name", collection_name)
        for var_name_b in graph.get_collection_ref(collection_name):
          trained_var_name_bs.add(var_name_b)

      var_names = [b.decode('utf-8') for b in trained_var_name_bs]
      vars = graph_query.find_variables_by_name(
          graph.get_collection_ref("variables"),
          var_names)
      graph_xform.replace_variable_initializers_with_current_values(
          graph,
          vars,
          "Trained")

    graph_execution.import_and_run_meta_graph(
      meta_graph_def=meta_graph_def,
      feed_dict={},
      result_pattern=re.compile(FLAGS.train_result_pattern),
      finish_session_fn=post_train,
    )
    meta_graph_def, _ = meta_graph.export_scoped_meta_graph()

  if FLAGS.test:
    graph_execution.import_and_run_meta_graph(
      meta_graph_def=meta_graph_def,
      feed_dict={},
      result_pattern=re.compile(FLAGS.test_result_pattern),
    )

  if meta_graph_def and FLAGS.output_file:
    eprint("meta_graph_def", [n.name for n in meta_graph_def.graph_def.node])
    graph_def = meta_graph_def.graph_def
    output_re = re.compile(FLAGS.output_result_pattern)
    output_node_names = ['py_funcs_json'] # HACK(adamb) So that pyfuncs still work.
    var_names = set()
    for n in graph_def.node:
      m = output_re.match(n.name)
      if not m:
        continue
      output_node_names.append(n.name)

      # If this isn't a function, then we're covered. Otherwise pick up needed
      # variables.
      if not m.group(2):
        continue

      # Look for collection of variable names referenced by this function.
      collection_name = "%s:variable_names" % m.group(1)
      function_var_name_bs = meta_graph_def.collection_def[collection_name].bytes_list.value
      for var_name_b in function_var_name_bs:
        # Remember the name of each variable referenced.
        var_names.add(var_name_b.decode('utf-8'))

    eprint("var_names", var_names)
    graph_xform.strip_meta_graph(meta_graph_def, output_node_names, var_names)

  if FLAGS.output_file:
    if FLAGS.output_format == "metagraph":
      graph_io.write_meta_graph_def(
        meta_graph_def=meta_graph_def,
        file=FLAGS.output_file,
        binary=FLAGS.output_binary)
    elif FLAGS.output_format == "graph":
      # If we trained and we're outputting a graph_def, we'll need to modify it.
      # We'll need to replace all the trained variables with the constants that
      # their initializers refer to.
      if FLAGS.train:
        pass
      graph_io.write_graph_def(
        graph_def=meta_graph_def.graph_def,
        file=FLAGS.output_file,
        binary=FLAGS.output_binary)

  feed_dict = {}
  # Properly find and strip prefix of constants, loading them with given prefix to feed_dict
  if FLAGS.feed_constants:
    feed_graph_def = graph_io.read_graph_def(FLAGS.feed_constants, FLAGS.feed_constants_binary)
    constants = graph_query.find_nodes_with_prefix(feed_graph_def, FLAGS.feed_constants_strip)
    constants_dict = graph_xform.constants_as_dict(constants)
    strip_prefix = FLAGS.feed_constants_strip
    add_prefix = FLAGS.feed_constants_prefix
    for name, value in constants_dict.items():
      if strip_prefix != None:
        if name.startswith(strip_prefix):
          name = name[len(strip_prefix):]
        else:
          continue
      feed_dict[add_prefix + name + ":0"] = value

  if FLAGS.run:
    results = graph_execution.import_and_run_meta_graph(
      meta_graph_def=meta_graph_def,
      feed_dict=feed_dict,
      result_pattern=re.compile(FLAGS.run_result_pattern),
    )

    graph_def = graph_xform.dict_as_graph_def(results)
    graph_io.write_graph_def(
      graph_def,
      file=FLAGS.result,
      binary=FLAGS.result_binary,
    )

  if FLAGS.repl:
    graph_repl.run()

if __name__ == '__main__':
  try:
    main()
  except Exception as ex:
    # TODO(adamb) Should do *real* error printing.
    # NOTE(adamb) Need to correlate expressions with line and character numbers!
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
