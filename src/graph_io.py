from tensorflow.core.framework import graph_pb2
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
