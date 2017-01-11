# Load graph, start a session, run it.

# vim: tabstop=2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import pprint
import re
import sys

import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow import gfile
from google.protobuf import text_format

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def read_graph_def(path, binary):
  mode = "r"
  if binary:
    mode += "b"

  graph_def = graph_pb2.GraphDef()
  with open(path, mode) as f:
    if binary:
      graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), graph_def)

  return graph_def

def find_nodes_with_pattern(graph_def, pattern):
  node_matches = []
  for n in graph_def.node:
    m = pattern.match(n.name)
    if m:
      node_matches.append((n, m))

  if len(node_matches) == 0:
    raise Exception("No nodes match pattern %s" % pattern)

  return node_matches

def find_results(graph_def, result_pattern):
  node_matches = find_nodes_with_pattern(graph_def, result_pattern)
  ops = [n.name + ":0" for n, m in node_matches]
  result_names = [m.group(1) for n, m in node_matches]

  return (result_names, ops)

def run_graph(graph_def, result_pattern, feed_dict):
  result_names, ops = find_results(graph_def, result_pattern)

  with tf.Session() as sess:
    ops = tf.import_graph_def(
      graph_def,
      return_elements=ops,
      name=""
    )

    tf.global_variables_initializer()
    result_tensors = sess.run(fetches=ops, feed_dict=feed_dict)

    return dict(zip(result_names, result_tensors))

def constants_as_dict(constants):
  d = {}
  for node in constants:
    name = node.name
    tensor = node.attr['value'].tensor
    value = None
    dtype = tensor.dtype
    if dtype == tf.bool:
      value = tensor.bool_val
    elif dtype == tf.float16:
      value = tensor.half_val
    elif dtype == tf.float32:
      value = tensor.float_val
    if dtype == tf.float64:
      value = tensor.double_val
    elif dtype == tf.complex64:
      value = tensor.scomplex_val
    elif dtype == tf.complex128:
      value = tensor.dcomplex_val
    elif dtype == tf.int64:
      value = tensor.int64_val
    elif dtype == tf.string:
      value = tensor.string_val

    d[name] = value

  return d

def dict_as_graph_def(constants_dict):
  with tf.Graph().as_default() as g:
    for name, value in constants_dict.items():
      tf.constant(value, name=name)

    return g.as_graph_def()

def write_graph_def(graph_def, file, binary):
  if binary:
    with open(file, "wb") as f:
      f.write(graph_def.SerializeToString())
  else:
    with open(file, "w") as f:
      data = text_format.MessageToString(graph_def)
      f.write(data)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("graphdef", nargs='?', type=str, default="/dev/stdin",
                      help="""Graph file to load.""")
  parser.add_argument("--binary-graphdef", nargs='?', type=bool, default=False,
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
  parser.add_argument("--result-prefix", nargs='?', type=str, default="main/",
                      help="""Prefix of nodes to read result from.""")
  parser.add_argument("--result-binary", nargs='?', type=bool, default=False,
                      help="""Whether or not to result in binary.""")
  parser.add_argument("--result", nargs='?', type=str, default="/dev/stdout")

  parser.add_argument("--test", nargs='?', type=bool, default=False,
                      help="""Run the tests graphs with given (or default) --test-* options""")
  parser.add_argument("--test-result-pattern", nargs='?', type=str, default="^test[^/]*/([^_].*)$",
                      help="""Pattern to discover test graph results.""")

  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.test == None:
    FLAGS.test = True

  if FLAGS.run == None:
    FLAGS.run = True

  feed_dict = {}
  # Properly find and strip prefix of constants, loading them with given prefix to feed_dict
  if FLAGS.feed_constants:
    feed_graph_def = read_graph_def(FLAGS.feed_constants, FLAGS.feed_constants_binary)
    constants = find_nodes_with_prefix(feed_graph_def, FLAGS.feed_constants_strip)
    constants_dict = constants_as_dict(constants)
    strip_prefix = FLAGS.feed_constants_strip
    add_prefix = FLAGS.feed_constants_prefix
    for name, value in constants_dict.items():
      if strip_prefix != None:
        if name.startswith(strip_prefix):
          name = name[len(strip_prefix):]
        else:
          continue
      feed_dict[add_prefix + name + ":0"] = value

  graph_def = read_graph_def(FLAGS.graphdef, FLAGS.binary_graphdef)

  if FLAGS.test:
    run_graph(
      graph_def=graph_def,
      feed_dict={},
      result_pattern=re.compile(FLAGS.test_result_pattern),
    )

  if FLAGS.run or not FLAGS.test:
    results = run_graph(
      graph_def=graph_def,
      feed_dict=feed_dict,
      result_pattern=re.compile("^%s([^_].*)$" % FLAGS.result_prefix),
    )

    graph_def = dict_as_graph_def(results)
    write_graph_def(
      graph_def,
      file=FLAGS.result,
      binary=FLAGS.result_binary,
    )
