from tensorflow.python.framework import dtypes

import graph_execution

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def AddWithStep(summary_protobuf, global_step) -> dtypes.int64:
  graph_execution._summary_writer.add_summary(summary_protobuf, global_step=global_step)
  return 0

def Add(summary_protobuf) -> dtypes.int64:
  graph_execution._summary_writer.add_summary(summary_protobuf)
  return 0
