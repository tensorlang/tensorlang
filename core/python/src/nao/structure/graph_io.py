from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import meta_graph

from google.protobuf import text_format

def _write_pb(pb, file, binary):
  if binary:
    with open(file, "wb") as f:
      f.write(pb.SerializeToString())
  else:
    with open(file, "w") as f:
      data = text_format.MessageToString(pb)
      f.write(data)

def _parse_pb(pb, data, binary):
  if binary:
    pb.ParseFromString(data)
  else:
    text_format.Merge(data, pb)
  return pb

def _read_pb(pb, file, binary):
  mode = "r"
  if binary:
    mode += "b"

  with open(file, mode) as f:
    return _parse_pb(pb, f.read(), binary)

def read_graph_def(file, binary):
  return _read_pb(graph_pb2.GraphDef(), file, binary)

def write_graph_def(graph_def, file, binary):
  _write_pb(graph_def, file, binary)

def read_meta_graph_def(file, binary):
  return _read_pb(meta_graph_pb2.MetaGraphDef(), file, binary)

def write_meta_graph_def(meta_graph_def, file, binary):
  _write_pb(meta_graph_def, file, binary)
