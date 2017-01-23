from tensorflow.python.framework import dtypes

import graph_execution

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def Add(summary_protobuf) -> dtypes.int64:
  graph_execution._summary_writer.add_summary(summary_protobuf)
  return 0
