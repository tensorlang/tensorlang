import tensorflow as tf

import re
import sys
import tensorflow as tf

from tensorflow.contrib.graph_editor import make_view
import tensorflow.contrib.graph_editor.transform as transform

from nao.structure import graph_io

from nao.compiler.retvalbag import RetvalBag

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class SubGraphViewFunction:
  def __init__(self, name, sgv, source_scope, inputs, output_names):
    eprint("sgv.graph", sgv.graph)
    self._nam = name
    self._inputs = inputs
    self._src_scope = source_scope
    self._output_names = output_names
    self._sgv = sgv

  def _name(self):
    return self._nam

  def apply(self, visitor, ctx, name, attrs, args):
    # We'll replace the inputs with the given args and copy all operations
    # (other than vars, which we leave as is) between outputs and inputs.

    g = tf.get_default_graph()
    full_scope = g.unique_name(self._name() or "sgv", False)
    scope_name = full_scope.split("/")[-1]
    eprint("full_scope", full_scope)
    eprint("scope_name", scope_name)
    replacements_ts = dict(zip(self._inputs, args))
    eprint("replacements_ts", replacements_ts)

    with tf.name_scope(None):
      self._copy_with_input_replacements(replacements_ts, full_scope)

    output_items = [(n, "%s/outputs/%s" % (full_scope, n)) for n in self._output_names]
    eprint("output_items", output_items)
    try:
      return RetvalBag(dict([(n, g.get_operation_by_name(p).outputs[0]) for n, p in output_items]))
    except KeyError as e:
      nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
      nodes.sort()
      eprint('error, but got nodes', nodes)
      raise e

  def _copy_with_input_replacements(self, replacement_ts, dst_scope):
    """Copy a subgraph, replacing some of its inputs.
    Note a replacement only happens if the tensor to be replaced
    is an input of the given subgraph. The inputs of a subgraph can
    be queried using sgv.inputs.
    Args:
      sgv: the source subgraph-view. This argument is converted to a subgraph
        using the same rules as the function subgraph.make_view.
      replacement_ts: dictionary mapping from original tensors to the
        replaced one.
      dst_scope: the destination scope.
      reuse_dst_scope: if True the dst_scope is re-used if it already exists.
        Otherwise, the scope is given a unique name based on the one given
        by appending an underscore followed by a digit (default).
    Raises:
      TypeError: if dst_graph is not a tf.Graph.
      StandardError: if sgv cannot be converted to a SubGraphView using
        the same rules as the function subgraph.make_view.
    """
    copier = transform.Transformer()
    # Replace tensor if possible.
    def replace_t_with_replacement_handler(info, t):
      if t in replacement_ts:
        eprint("Did find", t, "in", replacement_ts)
        return replacement_ts[t]
      else:
        eprint("Did not find", t, "in", replacement_ts)
        return transform.keep_t_if_possible_handler(info, t)
    copier.transform_external_input_handler = replace_t_with_replacement_handler

    orig_transform_op = copier.transform_op_handler
    def transform_op(info, op, copy_shape=True):
      eprint("transform_op", op.name)
      if isinstance(op, tf.Variable):
        eprint("Won't copy variable", op)
        return op

      return orig_transform_op(info, op, copy_shape)
    copier.transform_op_handler = transform_op
    dst_graph = tf.get_default_graph()
    eprint("dst_graph", dst_graph)

    return copier(
        self._sgv, dst_graph, dst_scope, self._src_scope, reuse_dst_scope=False)

class MetaGraphDefPackage:
  def __init__(self, meta_graph_def, import_path, internal_scope):
    eprint("import_path, internal_scope", import_path, internal_scope)

    internal_scope = internal_scope or import_path

    self._meta_graph_def = meta_graph_def
    self._internal_scope = internal_scope

    import_scope = import_path
    tensors = []
    functions = {} # name -> (scope, inputs, outputs)
    pattern = re.compile("%s/([^/]+)(?:$|(/inputs/|/outputs/).)" % internal_scope)
    graph_def = meta_graph_def.graph_def
    for n in graph_def.node:
      m = pattern.match(n.name)
      if not m:
        continue

      name = m.group(1)
      function_component = m.group(2)
      if not function_component:
        tensors.append(n.name)
        continue

      if name not in functions:
        function = (name, [], [])
        functions[name] = function
      else:
        function = functions[name]

      if function_component.startswith("/inputs/"):
        function[1].append(n.name)
      else:
        output_prefix_len = len(internal_scope) + len(name) + len(function_component) + 1
        function[2].append(n.name[output_prefix_len:])

    try:
      with tf.name_scope(None):
        tf.train.import_meta_graph(meta_graph_def, import_scope=import_path)
    except KeyError as e:
      nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
      nodes.sort()
      eprint('error, but got nodes', nodes)
      raise e

    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    nodes.sort()
    eprint('no error, but got nodes', nodes)

    g = tf.get_default_graph()
    exports = {} # name -> tensor | function
    for name in tensors:
      eprint("tensor", name, "%s/%s" % (import_scope, name))
      exports[name] = g.get_operation_by_name("%s/%s" % (import_scope, name))

    for name, (scope, full_input_names, output_names) in functions.items():
      # inputs = [g.get_tensor_by_name("%s/%s:0" % (import_scope, full_input_name)) for full_input_name in full_input_names]
      inputs = [g.get_tensor_by_name("%s/%s:0" % (import_scope, full_input_name)) for full_input_name in full_input_names]
      source_scope = "%s/%s/%s" % (import_path, internal_scope, scope)
      eprint("sgv_scope", "%s/%s/%s" % (import_path, internal_scope, scope), g)
      source_pattern = "%s/(?:_|outputs)/.*" % source_scope
      sgv = make_view(source_pattern, graph=g)
      eprint("sgv.inputs", list(sgv.inputs))
      exports[name] = SubGraphViewFunction(name, sgv, source_scope, inputs, output_names)

    self._exports = exports

  def apply(self, visitor, ctx, name, attrs, args):
    n, *_ = args

    try:
      return self._exports[n]
    except KeyError as e:
      eprint("only have exports", list(self._exports.keys()))
      raise e

def make_compile_fn(workspace, import_path, tags):
  basename, scope_name = (import_path + ":").split(":", 1)
  scope_name = scope_name[:-1]

  filepath = workspace.find_pkg_path(basename + ".metagraph.pbtxt")
  binary = False
  if filepath is None:
    return None

  def compile(resolved_imports, previous):
    eprint("_sf_tf_metagraph_package", import_path, scope_name)
    # TODO(adamb) how do we handle the fact that there may be multiple packages
    #     within the given file. Should we only parse out the one we want?
    meta_graph_def = graph_io.read_meta_graph_def(filepath, binary)
    return MetaGraphDefPackage(meta_graph_def, basename, scope_name)

  return ([], compile)
