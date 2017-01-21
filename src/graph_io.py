from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import meta_graph

from google.protobuf import text_format

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

def write_graph_def(graph_def, file, binary):
  if binary:
    with open(file, "wb") as f:
      f.write(graph_def.SerializeToString())
  else:
    with open(file, "w") as f:
      data = text_format.MessageToString(graph_def)
      f.write(data)

def read_meta_graph_def(path, binary):
  mode = "r"
  if binary:
    mode += "b"

  meta_graph_def = meta_graph_pb2.MetaGraphDef()
  with open(path, mode) as f:
    if binary:
      meta_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), meta_graph_def)

  return meta_graph_def

def write_meta_graph_def(meta_graph_def, file, binary):
  if binary:
    with open(file, "wb") as f:
      f.write(meta_graph_def.SerializeToString())
  else:
    with open(file, "w") as f:
      data = text_format.MessageToString(meta_graph_def)
      f.write(data)
