import json
import sys
import tensorflow as tf
import numpy

from tensorflow.python.framework import tensor_util

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def load_json(graph, const_name):
  json_tensor = None
  try:
    json_tensor = graph.get_tensor_by_name("%s:0" % const_name)
  except KeyError as e:
    pass

  if json_tensor is None:
    return None

  json_str = str(tensor_util.constant_value(json_tensor).astype('U'))
  return json.loads(json_str)

def store_json(const_name, value):
  json_str = json.dumps(value).encode('UTF-8')
  return tf.constant(json_str, name=const_name)
