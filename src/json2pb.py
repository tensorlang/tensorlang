# vim: tabstop=2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os.path
import pprint
import sys

import tensorflow as tf

from google.protobuf import text_format

pp = pprint.PrettyPrinter(indent=2, stream=sys.stderr)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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
    self.functions = {}

  def _tensor(self, ctx, *values):
    return values

  def _named_constant(self, ctx, name, shape, dtype, value):
    op = tf.constant(value, shape=shape, dtype=dtype, name=name)
    ctx[name] = op
    return op

  def _named_placeholder(self, ctx, name, shape, dtype):
    op = tf.placeholder(dtype, shape=shape, name=name)
    ctx[name] = op
    return op

  def _sf_type(self, ctx, name):
    return TopLevel.TYPES[name]

  def _sf_shape(self, ctx, dims):
    return tf.TensorShape(dims)

  def _sf_whole(self, ctx, digits):
    return int(digits)

  def _sf_fraction(self, ctx, decimal):
    return float(decimal)

  def _sf_local(self, ctx, name):
    return ctx[name];

  def _sf_graph(self, ctx, name, *exprs):
    with tf.variable_scope(name):
      emitted = []
      local_ops = {}
      with tf.variable_scope("impl"):
        # Accumulate ops to emit.
        for expr in exprs:
          if expr[0] == "_emit":
            name = expr[1]
            subexpr = expr[2]
            op = self.visit(local_ops, subexpr)
            local_ops[name] = op
            emitted.append(name)
          else:
            self.visit(local_ops, expr)

      with tf.variable_scope("emit"):
        for name in emitted:
          tf.identity(local_ops[name], name=name)

  def _sf_def_function(self, ctx, name, *rest):
    self.functions[name] = [name, *rest]

  def _sf_attrs(self, ctx):
    return {}

  def _sf_apply(self, ctx, ns_name, fn_name, attrs_expr, *arg_exprs):
    # attrs = self.visit(ctx, attrs_expr)
    args = [self.visit(ctx, expr) for expr in arg_exprs]

    if ns_name == None:
      # For now assume ns is tf if non-None.
      ns = tf
      # How to handle multiple return values?
      return getattr(ns, fn_name)(*args)
    else:
      function = this.functions[fn_name]
      # Need to start new scope
      # Need to preload locals with references to input operations
      # Need to visit expressions
      # Accumulate list of returned elements
      # Assign returned elements to locals, if appropriate

  def visit(self, ctx, expr):
    if type(expr) == list:
      expr_type = expr[0]
      attr = getattr(self, expr_type)
      eprint("visiting %s expr %s ctx: %s" % (expr_type, expr, ctx))

      if expr_type.startswith("_sf_"): # Special form
        result = attr(ctx, *expr[1:])
      elif expr_type.startswith("_named_"): # name, then expressions
        result = attr(ctx, expr[1], *[self.visit(ctx, subexpr) for subexpr in expr[2:]])
      else: # just expressions
        result = attr(ctx, *[self.visit(ctx, subexpr) for subexpr in expr[1:]])

      # eprint("visited %s expr %s => %s; ctx: %s" % (expr_type, expr, result, ctx))
      return result
    else:
      # eprint("visiting primitive %s ctx: %s" % (expr, ctx))
      return expr

def graph_def_from_exprs(exprs):
  with tf.Graph().as_default() as g:
    visitor = TopLevel()
    for expr in exprs:
      visitor.visit({}, expr)

    return g.as_graph_def()

def translate_graph(input_json, output_binary, output_graph):
  with open(input_json, "r") as f:
    s = f.read()
    input_exprs = json.loads(s)
    pp.pprint(input_exprs)

  output_graph_def = graph_def_from_exprs(input_exprs)

  if output_binary:
    with open(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
  else:
    with open(output_graph, "w") as f:
      data = text_format.MessageToString(output_graph_def)
      eprint("graph data:\n%s" % data)

      f.write(data)

  eprint("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_json", nargs='?', type=str, default="/dev/stdin",
    help="""JSON file to load.""")
  parser.add_argument("--output_binary", nargs='?', type=bool, default=False,
    help="""Whether or not to output in binary.""")
  parser.add_argument("--output_graph", nargs='?', type=str, default="/dev/stdout",
    help="""Path to write output in.""")

  FLAGS, unparsed = parser.parse_known_args()

  translate_graph(
    input_json=FLAGS.input_json,
    output_binary=FLAGS.output_binary,
    output_graph=FLAGS.output_graph,
  )
