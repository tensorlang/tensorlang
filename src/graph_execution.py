import graph_gen
import graph_io
import graph_query
import graph_xform
import json

from tensorflow.python.framework import meta_graph

import tensorflow as tf
import sys

from tensorflow.python.client import timeline

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def create_session():
  config = tf.ConfigProto(
    # log_device_placement=True,
    operation_timeout_in_ms=20000,
    inter_op_parallelism_threads=2,
  )

  return tf.Session(config=config)

import inspect

_summary_writer = None
def run_session(sess, result_pattern, feed_dict):
  global _summary_writer

  # HACK(adamb) Should parameterize this
  run_name = 'test'
  _summary_writer = tf.summary.FileWriter('./logdir/%s' % run_name, sess.graph)

  tf.global_variables_initializer().run()

  coord = tf.train.Coordinator()

  run_options = tf.RunOptions(
    # trace_level=tf.RunOptions.FULL_TRACE
  )
  run_metadata = tf.RunMetadata()

  threads = tf.train.start_queue_runners(coord=coord)

  try:
    result_names, ops = graph_query.find_results(sess.graph, result_pattern)
    result_tensors = sess.run(
      fetches=ops,
      feed_dict=feed_dict,
      options=run_options,
      run_metadata=run_metadata,
    )

    return dict(zip(result_names, result_tensors))
  finally:
    # Create the Timeline object, and write it to a json
    # tl = timeline.Timeline(run_metadata.step_stats)
    # ctf = tl.generate_chrome_trace_format()
    # with open('timeline.json', 'w') as f:
    #     f.write(ctf)

    coord.request_stop()
    coord.join(threads)

import graph_ffi
from tensorflow.python.ops import script_ops

def import_and_run_meta_graph(meta_graph_def, result_pattern, feed_dict):
  with create_session() as sess:
    meta_graph.import_scoped_meta_graph(
      meta_graph_def,
      input_map=None,
    )

    # NOTE(adamb) Could also store files to copy out in assets_collection
    js_py_func_data_tensor = None
    try:
      js_py_func_data_tensor = sess.graph.get_tensor_by_name("py_funcs_json:0")
    except KeyError as e:
      pass

    if js_py_func_data_tensor is not None:
      js_py_func_data = js_py_func_data_tensor.eval().decode('utf-8')
      py_func_data = json.loads(js_py_func_data)
      # eprint('loaded py_func_data', py_func_data)
      py_importer = graph_ffi.PythonImporter()
      py_importer.restore_py_funcs(script_ops._py_funcs, py_func_data)

    return run_session(sess, result_pattern, feed_dict)


def run_imported_graph(graph_def, result_pattern, feed_dict):
  with create_session() as sess:
    tf.import_graph_def(
      graph_def,
      name=""
    )

    return run_session(sess, result_pattern, feed_dict)
