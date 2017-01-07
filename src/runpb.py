# Load graph, start a session, run it.

# vim: tabstop=2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import pprint
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

def find_nodes_with_prefix(graph_def, prefix):
  nodes = [n for n in graph_def.node if n.name.startswith(prefix)]
  if len(nodes) == 0:
    raise Exception("No nodes with prefix %s" % prefix)

  return nodes

def find_outputs(graph_def, output_prefix):
  names = [n.name for n in find_nodes_with_prefix(graph_def, output_prefix)]

  ops = [name + ":0" for name in names]
  output_names = [name[len(output_prefix):] for name in names]

  return (names, ops)

def run_graph(graph_def, output_prefix, feed_dict):
  output_names, ops = find_outputs(graph_def, output_prefix)

  with tf.Session() as sess:
    ops = tf.import_graph_def(
      graph_def,
      return_elements=ops,
      name=""
    )

    tf.global_variables_initializer()
    output_tensors = sess.run(fetches=ops, feed_dict=feed_dict)

    return dict(zip(output_names, output_tensors))

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

def write_graph_def(graph_def, output_file, output_binary):
  if output_binary:
    with open(output_file, "wb") as f:
      f.write(graph_def.SerializeToString())
  else:
    with open(output_file, "w") as f:
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
  parser.add_argument("--output-prefix", nargs='?', type=str, default="main/emit/",
                      help="""Prefix of nodes to read output from.""")
  parser.add_argument("--output-binary", nargs='?', type=bool, default=False,
                      help="""Whether or not to output in binary.""")
  parser.add_argument("--output", nargs='?', type=str, default="/dev/stdout")

  FLAGS, unparsed = parser.parse_known_args()

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
  outputs = run_graph(
    graph_def=graph_def,
    feed_dict=feed_dict,
    output_prefix=FLAGS.output_prefix,
  )

  graph_def = dict_as_graph_def(outputs)
  write_graph_def(
    graph_def,
    output_file=FLAGS.output,
    output_binary=FLAGS.output_binary,
  )
