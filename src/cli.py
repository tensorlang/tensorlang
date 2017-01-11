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

class Context:
  def __init__(self):
    self.locals = {}
    self.aliases = {}
    self.root_suffixes = {}
    self._leaves = set()

  def set_alias(self, name, retvals):
    self.aliases[name] = retvals

  def get_alias(self, name):
    return self.aliases[name]

  def set_local(self, name, value):
    self.locals[name] = value

  def get_local(self, name):
    return self.locals[name]

  def possible_leaf(self, op):
    self._leaves.add(op)

  def eliminate_leaf(self, op):
    if type(op) == tf.Tensor:
      self._leaves.discard(op)

  def leaves(self):
    return frozenset(self._leaves)

  def unique_name(self, root):
    if not root in self.root_suffixes:
      self.root_suffixes[root] = -1

    suffix = self.root_suffixes[root]
    suffix = suffix + 1
    self.root_suffixes[root] = suffix

    return "%s_%s" % (root, suffix)

  def __str__(self):
    return "%s" % self.locals


class TopLevel:
  TYPES = {
	  "float": tf.float32,
	  "double": tf.float64,
	  "int8": tf.int8,
	  "int16": tf.int16,
	  "int32": tf.int32,
	  "int64": tf.int64,
	  "uint8": tf.uint8,
	  "uint16": tf.uint16,
	  "string": tf.string,
	  "bool": tf.bool,
	  "complex64": tf.complex64,
	  "complex128": tf.complex128,
	  "qint8": tf.qint8,
	  "qint32": tf.qint32,
	  "quint": tf.quint8,
  }

  def __init__(self):
    self.nesting_level = 0
    self.functions = {}

  # "primitive" values
  def _sf_type(self, ctx, name):
    return TopLevel.TYPES[name]

  def _sf_shape(self, ctx, dims):
    return tf.TensorShape(dims)

  def _sf_whole(self, ctx, digits):
    return int(digits)

  def _sf_fraction(self, ctx, decimal):
    return float(decimal)

  def _sf_list(self, ctx, *exprs):
    return [self.visit(ctx, expr) for expr in exprs]

  def _named_tensor(self, ctx, name, shape, dtype, value):
    op = tf.constant(value, shape=shape, dtype=dtype, name=name)
    ctx.possible_leaf(op)

    if name != None:
      ctx.set_local(name, op)
    return op

  def _named_placeholder(self, ctx, name, shape, dtype):
    op = tf.placeholder(dtype, shape=shape, name=name)
    ctx.set_local(name, op)
    return op

  # applying a function
  def _sf_apply(self, ctx, name, ns_name, fn_name, attrs_expr, *arg_exprs):
    # attrs = self.visit(ctx, attrs_expr)
    # args = [self.visit(ctx, expr) for expr in arg_exprs]
    args = []
    for expr in arg_exprs:
      arg = self.visit(ctx, expr)
      ctx.eliminate_leaf(arg)
    #   eprint("arg %s -> %s" % (expr, arg))
      args.append(arg)

    if ns_name != None:
      # For now assume ns is tf if non-None.
      ns = tf
      # How to handle multiple return values?
    #   eprint("tf.%s(%s)" % (fn_name, args))
      result = getattr(ns, fn_name)(*args)
      ctx.possible_leaf(result)
      if name != None:
        ctx.set_local(name, result)

      return result

    else:
      function = self.functions[fn_name]
      scope_name = name
      if scope_name == None:
        scope_name = ctx.unique_name(fn_name)

      returned = []
      new_ctx = Context()
      with tf.variable_scope(scope_name):
        arg_specs, retval_specs, *body = function[1:]

        # preload locals with references to input operations
        for arg, arg_spec in zip(args, arg_specs):
          arg_name = arg_spec[0]
          new_ctx.set_local(arg_name, arg)

        # Need to visit expressions
        for expr in body:
          self.visit(new_ctx, expr)

      for retval_name, retval_argname in retval_specs:
        returned.append((retval_name, new_ctx.get_local(retval_argname)))

      # For now we only use the first retval
      result = returned[0][1]
      if name != None:
        ctx.possible_leaf(result)
        ctx.set_local(name, result)

      return result

  def _sf_local(self, ctx, name):
    # eprint(ctx)
    return ctx.get_local(name)

  # generating graphs directly
  def visit_graph_exprs(self, ctx, retval_names, exprs):
    for expr in exprs:
      if expr[0] == "__retval":
        name = expr[1]
        subexpr = expr[2]
        op = self.visit(ctx, subexpr)
        ctx.set_local(name, op)
        retval_names.append(name)
      elif expr[0] == "__sf_after_leaves":
        # TODO(adamb) Should actually nest local variables AND leaves
        after_exprs = expr[1:]
        leaves = ctx.leaves()
        with tf.control_dependencies(leaves):
          self.visit_graph_exprs(ctx, retval_names, after_exprs)
      else:
        self.visit(ctx, expr)

  def _sf_graph(self, ctx, name, *exprs):
    with tf.variable_scope(name):
      retval_names = []
      local_ops = Context()

      with tf.variable_scope("_"):
        self.visit_graph_exprs(local_ops, retval_names, exprs)

      for retval_name in retval_names:
        op = local_ops.get_local(retval_name)
        tf.identity(op, name=retval_name)

  def _sf_def_function(self, ctx, name, *rest):
    self.functions[name] = [name, *rest]

  def _sf_attrs(self, ctx):
    return {}

  def visit(self, ctx, expr):
    self.nesting_level = self.nesting_level + 1
    if type(expr) == list:
      expr_type = expr[0]
    #   eprint("%s%s" % ('  ' * self.nesting_level, expr))
      attr = getattr(self, expr_type)

      if expr_type.startswith("_sf_"): # Special form
        result = attr(ctx, *expr[1:])
      elif expr_type.startswith("_named_"): # name, then expressions
        result = attr(ctx, expr[1], *[self.visit(ctx, subexpr) for subexpr in expr[2:]])
      else: # just expressions
        result = attr(ctx, *[self.visit(ctx, subexpr) for subexpr in expr[1:]])

      # eprint("visited %s expr %s => %s; ctx: %s" % (expr_type, expr, result, ctx))
      self.nesting_level = self.nesting_level - 1
      return result
    else:
      # eprint("visiting primitive %s ctx: %s" % (expr, ctx))
      self.nesting_level = self.nesting_level - 1
      return expr

def graph_def_from_exprs(exprs):
  with tf.Graph().as_default() as g:
    visitor = TopLevel()
    for expr in exprs:
      visitor.visit(Context(), expr)

    return g.as_graph_def()

def compile_graph(input_json):
  with open(input_json, "r") as f:
    s = f.read()
    input_exprs = json.loads(s)
    # pp.pprint(input_exprs)

  return graph_def_from_exprs(input_exprs)

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
    graph_def = read_graph_def(FLAGS.graphdef, FLAGS.binary_graphdef)

  if FLAGS.input_json:
    graph_def = compile_graph(FLAGS.input_json)

  if FLAGS.output_graph:
    write_graph_def(
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

  if FLAGS.test:
    run_graph(
      graph_def=graph_def,
      feed_dict={},
      result_pattern=re.compile(FLAGS.test_result_pattern),
    )

  if FLAGS.run:
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
