import graph_gen
import graph_io
import graph_query
import graph_xform

import tensorflow as tf

def run(graph_def, result_pattern, feed_dict):
  result_names, ops = graph_query.find_results(graph_def, result_pattern)

  with tf.Session() as sess:
    ops = tf.import_graph_def(
      graph_def,
      return_elements=ops,
      name=""
    )

    tf.global_variables_initializer()
    result_tensors = sess.run(fetches=ops, feed_dict=feed_dict)

    return dict(zip(result_names, result_tensors))
