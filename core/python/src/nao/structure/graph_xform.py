import sys

import tensorflow as tf

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.framework import graph_util

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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

def replace_variable_initializers_with_current_values(graph, vars, value_suffix):
  with graph.as_default():
    for var in vars:
      var_op_name = var.op.name
      var_value = var.value().eval()
      var_init_op = tf.assign(
        var,
        tf.constant(var_value, name="%s/%s" % (var_op_name, value_suffix)),
        name="%s/Assign%s" % (var_op_name, value_suffix)).op
      var._initializer_op = var_init_op
      eprint("Resetting initializer for var", var)

def strip_meta_graph(meta_graph_def, node_names, var_names):
  node_names = node_names[:]
  collections = meta_graph_def.collection_def

  # Look for matching variable names and initializers and keep them too.
  var_def = variable_pb2.VariableDef()
  for var_col_name in ["variables", "trainable_variables"]:
    var_def_bs = collections[var_col_name].bytes_list.value
    for var_def_b in var_def_bs:
      var_def.ParseFromString(var_def_b)
      if var_def.variable_name not in var_names:
        # TODO(adamb) Should remove variable from collection.
        continue
      node_names.append(var_def.initializer_name)

  wc_def = control_flow_pb2.WhileContextDef()
  wc_values = collections["while_context"].bytes_list.value
  for wc_ix in range(len(wc_values) - 1, -1, -1):
    wc_bytes = wc_values[wc_ix]
    wc_def.ParseFromString(wc_bytes)
    unused = True
    wc_pivot_name = wc_def.pivot_name
    for name in node_names:
      if name.startswith(wc_pivot_name):
        unused = False
        break

    if unused:
      del wc_values[wc_ix]

  graph_def = meta_graph_def.graph_def
  eprint("only keeping", node_names, "from", [n.name for n in graph_def.node])
  graph_def = graph_util.extract_sub_graph(graph_def, node_names)
  meta_graph_def.graph_def.CopyFrom(graph_def)
