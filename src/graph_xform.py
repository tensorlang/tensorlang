import tensorflow as tf

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
