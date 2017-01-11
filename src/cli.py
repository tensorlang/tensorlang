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

import graph_gen
import graph_io
import graph_query
import graph_xform
import graph_execution

import tensorflow as tf

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def compile_graph(input_json):
  with open(input_json, "r") as f:
    s = f.read()
    input_exprs = json.loads(s)
    # pp.pprint(input_exprs)

  return graph_gen.graph_def_from_exprs(input_exprs)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("graphdef", nargs='?', type=str,
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

  parser.add_argument("--input-json", nargs='?', type=str,
                      help="""JSON file to load.""")
  parser.add_argument("--output-binary", nargs='?', type=bool, default=False,
                      help="""Whether or not to output in binary.""")
  parser.add_argument("--output-graph", nargs='?', type=str,
                      help="""Path to write output in.""")

  FLAGS, unparsed = parser.parse_known_args()

  graph_def = None

  if FLAGS.graphdef:
    graph_def = graph_io.read_graph_def(FLAGS.graphdef, FLAGS.binary_graphdef)

  if FLAGS.input_json:
    graph_def = compile_graph(FLAGS.input_json)

  if FLAGS.output_graph:
    graph_io.write_graph_def(
      graph_def=graph_def,
      file=FLAGS.output_graph,
      binary=FLAGS.output_binary)

  if FLAGS.test == None:
    FLAGS.test = True

  if FLAGS.run == None:
    FLAGS.run = True

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

  if FLAGS.test:
    graph_execution.run(
      graph_def=graph_def,
      feed_dict={},
      result_pattern=re.compile(FLAGS.test_result_pattern),
    )

  if FLAGS.run:
    results = graph_execution.run(
      graph_def=graph_def,
      feed_dict=feed_dict,
      result_pattern=re.compile("^%s([^_].*)$" % FLAGS.result_prefix),
    )

    graph_def = dict_as_graph_def(results)
    graph_io.write_graph_def(
      graph_def,
      file=FLAGS.result,
      binary=FLAGS.result_binary,
    )
