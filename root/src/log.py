from tensorflow.python.framework import dtypes

import sys

from nao.run import graph_summary

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def LogWithStep(summary_protobuf, global_step) -> dtypes.int64:
  graph_summary.get_summary_writer().add_summary(summary_protobuf, global_step=global_step)
  return 0

def Log(summary_protobuf) -> dtypes.int64:
  eprint("Writing summary!")
  graph_summary.get_summary_writer().add_summary(summary_protobuf)
  return 0

def Debug(x) -> dtypes.int64:
  eprint("Debug %s" % x)
  return 0
