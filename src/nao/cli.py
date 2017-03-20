from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime

import argparse
import json
import pprint
import re
import sys
import traceback
import gc

from os import path

from nao import graph_gen
from nao import graph_io
from nao import graph_query
from nao import graph_xform
from nao import graph_repl
from nao import graph_execution
from nao import parser as source_parser

import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import control_flow_pb2

import subprocess

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

pp = pprint.PrettyPrinter(indent=2, stream=sys.stderr).pprint

def compile_meta_graph(input_json):
  with open(input_json, "r") as f:
    s = f.read()
    input_exprs = json.loads(s)
    # pp.pprint(input_exprs)

    return graph_gen.meta_graph_def_from_exprs(input_exprs)

def parse_packages(src_root, pkg_root, package_names):
  p = source_parser.PalletParser(src_root, pkg_root)

  for package_name in package_names:
    p.resolve_import(package_name, None)

  return p.pallet()

def parse_source(src_root, pkg_root, package_name, source):
  p = source_parser.PalletParser(src_root, pkg_root)
  p.put_source(package_name, source)
  p.resolve_import(package_name, None)

  return p.pallet()

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("package_names", type=str, nargs='*')
  parser.add_argument("--root", type=str,
                      help="""Specify root directory to search for imports from""")
  parser.add_argument("--source", type=str,
                      help="""Specify source code instead of reading from file""")

  parser.add_argument("--reopen-stderr")
  parser.add_argument("--reopen-stdout")

  parser.add_argument("--metagraphdef", type=str,
                      help="""Graph file to load.""")
  parser.add_argument("--binary-metagraphdef", type=bool, default=False,
                      help="""Whether or not input is binary.""")
  parser.add_argument("--feed-constants", type=str,
                      help="""Path to GraphDef protobuf with constants to feed""")
  parser.add_argument("--feed-constants-strip", nargs='?', type=str, default="",
                      help="""Prefix to filter for (and strip from) constants""")
  parser.add_argument("--feed-constants-prefix", type=str,
                      help="""Prefix to add to constant names in feed""")
  parser.add_argument("--feed-constants-binary", type=bool, default=False,
                      help="""Whether or not feed constant protobuf is binary""")

  parser.add_argument("--run", default=False, action='store_const', const=True,
                      help="""Run the graph with given (or default) --result* and --feed-* options""")
  parser.add_argument("--run-result-pattern", type=str, default="^(${package}/Main)/outputs/(.*)$",
                      help="""Pattern to discover run results.""")
  parser.add_argument("--result-binary", default=False, action='store_const', const=True,
                      help="""Whether or not to result in binary.""")
  parser.add_argument("--result", type=str, default="/dev/stdout")

  parser.add_argument("--test", default=False, action='store_const', const=True,
                      help="""Run the tests graphs with given (or default) --test-* options""")
  parser.add_argument("--test-result-pattern", type=str, default="^(${package}/Test[^/]*)/outputs/(.*)$",
                      help="""Pattern to discover test graph results.""")

  parser.add_argument("--repl", default=False, action='store_const', const=True,
                      help="""Start REPL""")

  parser.add_argument("--tensorboard", nargs='?', default="",
                      help="""Start tensorboard server on the given IP:PORT, with the given --log-root or --log-dir""")

  parser.add_argument("--jupyter-kernel", nargs='?', default="",
                      help="""Start Jupyter kernel with the given configuration file""")

  parser.add_argument("--train", default=False, action='store_const', const=True,
                      help="""Run train graphs with given (or default) --train-* options""")
  parser.add_argument("--train-result-pattern", type=str, default="^(${package}/Train[^/]*)/outputs/(.*)$",
                      help="""Pattern to discover train graph results.""")

  parser.add_argument("--input-json", type=str,
                      help="""JSON file to load.""")

  parser.add_argument("--workspace",
                      help="""Default value for workspace""")
  parser.add_argument("--log-root", type=str,
                      help="""Which directory to calculate default log dir from.""")
  parser.add_argument("--log-dir", type=str,
                      help="""Which directory to put logs in.""")

  parser.add_argument("--output", default=False, action='store_const', const=True,
                      help="""Output graph""")
  parser.add_argument("--output-root", type=str,
                      help="""When automatically constructing output path, use this as base""")
  parser.add_argument("--output-name", type=str,
                      help="""Base name to use for output file name. Defaults to ${package} if there's only one.""")
  parser.add_argument("--output-result-pattern", type=str, default="^(${package}/[^/]*)(/outputs/[^/]*)?$",
                      help="""Pattern to discover outputs of graph to output.""")
  parser.add_argument("--output-format", type=str, default="metagraph",
                      help="""Defaults to metagraph""")
  parser.add_argument("--output-binary", default=False, action='store_const', const=True,
                      help="""Whether or not to output in binary.""")
  parser.add_argument("--output-file", type=str,
                      help="""Path to write output to. Defaults to ${output-name}.${output-format}""")

  FLAGS = parser.parse_args()

  if FLAGS.reopen_stderr:
    stderr_file = open(FLAGS.reopen_stderr, 'a')
    os.close(sys.stderr.fileno())
    os.dup2(stderr_file.fileno(), sys.stderr.fileno())

  if FLAGS.reopen_stdout:
    stdout_file = open(FLAGS.reopen_stdout, 'a')
    os.close(sys.stdout.fileno())
    os.dup2(stdout_file.fileno(), sys.stdout.fileno())


  package_names = FLAGS.package_names

  should_parse = len(package_names) > 0 or FLAGS.source
  if not (should_parse or FLAGS.run or FLAGS.test or FLAGS.output):
    if os.isatty(1):
      FLAGS.repl = True

  if should_parse and not (FLAGS.repl or FLAGS.run or FLAGS.test or FLAGS.output):
    FLAGS.output = True

  def search_upwards(startdir, filename):
    curdir = startdir
    while True:
      if path.exists(path.join(curdir, filename)):
        return curdir
      lastdir = curdir
      curdir = path.dirname(curdir)
      if curdir == lastdir:
        return None

  if not FLAGS.workspace:
    FLAGS.workspace = os.environ.get("NAOPATH", "")
    if not FLAGS.workspace:
      FLAGS.workspace = search_upwards(os.getcwd(), ".naopath")
    if not FLAGS.workspace:
      FLAGS.workspace = "."

  if FLAGS.output_root is None:
    FLAGS.output_root = path.join(FLAGS.workspace, "pkg")

  if FLAGS.root is None:
    FLAGS.root = path.join(FLAGS.workspace, "src")

  if FLAGS.log_root is None:
    FLAGS.log_root = path.join(FLAGS.workspace, "log")

  if FLAGS.tensorboard is None:
    FLAGS.tensorboard = "127.0.0.1:6006"

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
  eprint("FLAGS", FLAGS)
  eprint("package_names", package_names)

  if FLAGS.tensorboard != "":
    tb_host, tb_port = FLAGS.tensorboard.split(':', 2)
    tb_logdir = FLAGS.log_dir or FLAGS.log_root
    if tb_port is not None:
      tb_port = int(tb_port)

    from nao import tensorboard_server
    exit(tensorboard_server.main(tb_logdir, tb_host=tb_host, tb_port=tb_port))

  if FLAGS.jupyter_kernel != "":
    jupyter_config_file = FLAGS.jupyter_kernel
    from nao import jupyter_kernel, jupyter_kernel_driver

    if jupyter_config_file:
      eprint("Reading jupyter_config file '%s'..." % jupyter_config_file)
      jupyter_config = json.loads("".join(open(jupyter_config_file).readlines()))
    else:
      jupyter_config = {
        'control_port'      : 0,
        'hb_port'           : 0,
        'iopub_port'        : 0,
        'ip'                : '127.0.0.1',
        'key'               : str(uuid.uuid4()),
        'shell_port'        : 0,
        'signature_scheme'  : 'hmac-sha256',
        'stdin_port'        : 0,
        'transport'         : 'tcp'
      }

    pallet_parser = source_parser.PalletParser(FLAGS.root, FLAGS.output_root)
    repl_session = graph_repl.ReplSession(pallet_parser)
    driver = jupyter_kernel_driver.Driver(repl_session)
    exit(jupyter_kernel.Kernel(jupyter_config, driver.info(), driver.do).run())

  def log_dir_fn(pkg_names):
    if FLAGS.log_dir:
      return FLAGS.log_dir

    # HACK(adamb) Should parameterize this
    run_name = datetime.datetime.utcnow().strftime("%F_%H-%M-%S")
    return path.join(FLAGS.log_root, pkg_names[0], run_name)

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
      eprint("saving vars", var_names, vars)
      graph_xform.replace_variable_initializers_with_current_values(
          graph,
          vars,
          "Trained")

    graph_execution.import_and_run_meta_graph(
      meta_graph_def=meta_graph_def,
      feed_dict={},
      result_pattern=re.compile(FLAGS.train_result_pattern),
      finish_session_fn=post_train,
      log_dir_fn=log_dir_fn,
    )
    meta_graph_def, _ = meta_graph.export_scoped_meta_graph()

  if FLAGS.test:
    graph_execution.import_and_run_meta_graph(
      meta_graph_def=meta_graph_def,
      feed_dict={},
      log_dir_fn=log_dir_fn,
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
      eprint("collection_name", collection_name)
      function_var_name_bs = meta_graph_def.collection_def[collection_name].bytes_list.value
      for var_name_b in function_var_name_bs:
        # Remember the name of each variable referenced.
        var_names.add(var_name_b.decode('utf-8'))

    eprint("var_names", var_names)
    eprint("output_node_names", output_node_names)
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
      log_dir_fn=log_dir_fn,
      result_pattern=re.compile(FLAGS.run_result_pattern),
    )

    graph_def = graph_xform.dict_as_graph_def(results)
    graph_io.write_graph_def(
      graph_def,
      file=FLAGS.result,
      binary=FLAGS.result_binary,
    )

  if FLAGS.repl:
    graph_repl.run(source_parser.PalletParser(FLAGS.root, FLAGS.output_root))

if __name__ == '__main__':
  try:
    main()
    # with tf.device("/cpu:0"):
    #   main()
  except Exception as ex:
    # TODO(adamb) Should do *real* error printing.
    # NOTE(adamb) Need to correlate expressions with line and character numbers!
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
