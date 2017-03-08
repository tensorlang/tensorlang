from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

if "" == os.getenv("NAO_DID_EXEC", ""):
  CUDA_HOME = "/usr/local/cuda"
  DYLD_LIBRARY_PATH = os.getenv("DYLD_LIBRARY_PATH", '')
  PATH = os.getenv("PATH", '')
  env = {
    "NAO_DID_EXEC": "1",
    "CUDA_HOME": CUDA_HOME,
    "DYLD_LIBRARY_PATH": str.join(":", [CUDA_HOME + "/lib", CUDA_HOME + "/extras/CUPTI/lib", DYLD_LIBRARY_PATH]),
    "PATH": str.join(":", [CUDA_HOME + "/bin", PATH]),
  }
  args = [sys.executable, *sys.argv]
  print("args", args)
  os.execve(sys.executable, args, env)

import argparse
import json
import pprint
import re
import sys
import traceback
import gc

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

def parse_packages(root, package_names):
  with subprocess.Popen(
      ["node", "lib/cli.js", "--root", root, "--parse=-", *package_names],
      stdout=subprocess.PIPE) as proc:
    expr_text = proc.stdout.read().decode('utf-8')
    expr = json.loads(expr_text)
  return expr

def parse_source(root, source):
  with subprocess.Popen(
      ["node", "lib/cli.js", "--root", root, "--source", source, "--parse=-"],
      stdout=subprocess.PIPE) as proc:
    expr_text = proc.stdout.read().decode('utf-8')
  return json.loads(expr_text)

def main():
  parser = argparse.ArgumentParser()

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

  parser.add_argument("--run", nargs='?', type=bool, default=False,
                      help="""Run the graph with given (or default) --result* and --feed-* options""")
  parser.add_argument("--run-result-pattern", nargs='?', type=str, default="^${package}/([^_].*)$",
                      help="""Pattern to discover run results.""")
  parser.add_argument("--result-binary", nargs='?', type=bool, default=False,
                      help="""Whether or not to result in binary.""")
  parser.add_argument("--result", nargs='?', type=str, default="/dev/stdout")

  parser.add_argument("--test", nargs='?', type=bool, default=False,
                      help="""Run the tests graphs with given (or default) --test-* options""")
  parser.add_argument("--test-result-pattern", nargs='?', type=str, default="^(${package}/test[^/]*)/([^_].*)$",
                      help="""Pattern to discover test graph results.""")

  parser.add_argument("--repl", nargs='?', type=bool, default=False,
                      help="""Start REPL""")

  parser.add_argument("--train", nargs='?', type=bool, default=False,
                      help="""Run train graphs with given (or default) --train-* options""")
  parser.add_argument("--train-result-pattern", nargs='?', type=str, default="^(${package}/train[^/]*)/([^_].*)$",
                      help="""Pattern to discover train graph results.""")

  parser.add_argument("--input-json", nargs='?', type=str,
                      help="""JSON file to load.""")

  parser.add_argument("--output-binary", nargs='?', type=bool, default=False,
                      help="""Whether or not to output in binary.""")
  parser.add_argument("--output-metagraphdef", nargs='?', type=str,
                      help="""Path to write output in.""")
  parser.add_argument("--output-graphdef", nargs='?', type=str,
                      help="""Path to write output in.""")
  parser.add_argument("--output-result-pattern", nargs='?', type=str, default="^(${package}/[^/]*)(/outputs/[^/]*)?$",
                      help="""Pattern to discover run results.""")

  FLAGS, package_names = parser.parse_known_args(args=sys.argv[1:])

  if FLAGS.test == None:
    FLAGS.test = True

  if FLAGS.run == None:
    FLAGS.run = True

  if FLAGS.repl == None:
    FLAGS.repl = True

  if FLAGS.train == None:
    FLAGS.train = True

  if not (FLAGS.run or FLAGS.test or FLAGS.output_metagraphdef or FLAGS.output_graphdef):
    if os.isatty(1):
      FLAGS.repl = True


  meta_graph_def = None

  if FLAGS.metagraphdef:
    meta_graph_def = graph_io.read_meta_graph_def(FLAGS.metagraphdef, FLAGS.binary_metagraphdef)

  if FLAGS.input_json:
    meta_graph_def = compile_meta_graph(FLAGS.input_json)

  if len(package_names) > 0 or FLAGS.source:
    if FLAGS.source:
      package_names = ["main"]
      expressions = parse_source(FLAGS.root, FLAGS.source)
    else:
      expressions = parse_packages(FLAGS.root, package_names)

    # print("parsed", expressions)
    meta_graph_def = graph_gen.meta_graph_def_from_exprs(expressions)
    # We need to do this so we clean up references to py_funcs. LAME.
    gc.collect()

  # Now that we know our package names, use them to target the proper results.
  package_pattern = "(?:" + str.join("|", package_names) + ")"
  FLAGS.test_result_pattern = FLAGS.test_result_pattern.replace("${package}", package_pattern)
  FLAGS.train_result_pattern = FLAGS.train_result_pattern.replace("${package}", package_pattern)
  FLAGS.run_result_pattern = FLAGS.run_result_pattern.replace("${package}", package_pattern)
  FLAGS.output_result_pattern = FLAGS.output_result_pattern.replace("${package}", package_pattern)

  if FLAGS.train:
    def post_train(session, result_scope_prefixes):
      graph = session.graph

      vars_by_name = {}
      for var in graph.get_collection_ref("variables"):
        vars_by_name[var.name] = var

      trained_var_name_bs = set()
      for result_scope_prefix in result_scope_prefixes:
        collection_name = "%s:variable_names" % result_scope_prefix
        eprint("collection_name", collection_name)
        for var_name_b in graph.get_collection_ref(collection_name):
          trained_var_name_bs.add(var_name_b)

      with graph.as_default():
        for var_name_b in trained_var_name_bs:
          var_name = var_name_b.decode('utf-8')
          var = vars_by_name[var_name]
          var_op_name = var.op.name
          var_value = var.value().eval()
          var_init_op = tf.assign(
            var,
            tf.constant(var_value, name="%s/Trained" % var_op_name),
            name="%s/AssignTrained" % var_op_name).op
          var._initializer_op = var_init_op
          eprint("var:", var, var.initializer.name, var.to_proto())

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

  if meta_graph_def and FLAGS.output_result_pattern:
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
      for var_name_b in meta_graph_def.collection_def[collection_name].bytes_list.value:
        # Remember the name of each variable referenced.
        var_names.add(var_name_b.decode('utf-8'))

    eprint("var_names", var_names)
    var_def = variable_pb2.VariableDef()
    # Look for matching variable names and initializers and keep them too.
    for var_col_name in ["variables", "trainable_variables"]:
      var_def_bs = meta_graph_def.collection_def[var_col_name].bytes_list.value
      for var_def_b in var_def_bs:
        var_def.ParseFromString(var_def_b)
        if var_def.variable_name not in var_names:
          continue
        output_node_names.append(var_def.initializer_name)

    wc_def = control_flow_pb2.WhileContextDef()
    wc_values = meta_graph_def.collection_def["while_context"].bytes_list.value
    for wc_ix in range(len(wc_values) - 1, -1, -1):
      wc_bytes = wc_values[wc_ix]
      wc_def.ParseFromString(wc_bytes)
      unused = True
      wc_pivot_name = wc_def.pivot_name
      for name in output_node_names:
        if name.startswith(wc_pivot_name):
          unused = False
          break

      if unused:
        eprint("removing unused context", wc_def.pivot_name)
        del wc_values[wc_ix]


    graph_def = graph_util.extract_sub_graph(graph_def, output_node_names)
    meta_graph_def.graph_def.CopyFrom(graph_def)

  if FLAGS.output_metagraphdef:
    graph_io.write_meta_graph_def(
      meta_graph_def=meta_graph_def,
      file=FLAGS.output_metagraphdef,
      binary=FLAGS.output_binary)

  if FLAGS.output_graphdef:
    graph_io.write_graph_def(
      graph_def=meta_graph_def.graph_def,
      file=FLAGS.output_graphdef,
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
