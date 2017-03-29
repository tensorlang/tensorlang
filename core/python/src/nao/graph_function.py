import inspect
import re
import sys

import tensorflow as tf
from tensorflow.contrib.graph_editor import make_view
import tensorflow.contrib.graph_editor.transform as transform

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class RetvalBag:
  def __init__(self, a_dict):
    self._d = {}

    for k, v in a_dict.items():
      if type(v) == RetvalBag:
        v = v.get(None)
      self._d[k] = v

  def get(self, key):
    if key == None:
      key = self._default_key()
    return self._d[key]

  def values(self):
    return self._d.values()

  def __str__(self):
    return "RetvalBag(%s)" % self._d

  def _default_key(self):
    l = len(self._d)
    if l == 0:
      raise Exception("Can't get default retval for an empty RetvalBag")
    if l > 1:
      raise Exception("Can't get default retval a RetvalBag with more than one entry: %s" % self._d)
    return list(self._d.keys())[0]


class Package:
  def __init__(self, ctx):
    self._ctx = ctx

  def ctx(self):
    return self._ctx

  def apply(self, visitor, ctx, name, attrs, args):
    n, *_ = args
    if self._ctx.has_attr(n):
      value = self._ctx.get_attr(n)
    else:
      value = self._ctx.get_local_strict(n)

    if not n[0].isupper():
      raise Exception("Tried to use non-exported value named: %s" % n)

    if name:
      return tf.identity(value, name=name)

    return value

class PythonPackage:
  def __init__(self, mod, prepend_with_context=False):
    self._mod = mod
    self._prepend_with_context = prepend_with_context

  def apply(self, visitor, ctx, name, attrs, args):
    # eprint("applying %s to args %s" % (self, args))
    n, *_ = args
    if n.startswith('__'):
      raise Exception("Tried to use non-exported namespace entry named: %s" % n)

    val = getattr(self._mod, n)
    if callable(val):
      return PrimitiveFunction(val, self._prepend_with_context)

    return val


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

class PrimitiveFunction:
  def __init__(self, fn, prepend_with_context=False):
    self._prepend_with_context = prepend_with_context
    self._fn = fn
    sig = inspect.signature(fn)
    self._params = sig.parameters

    self._name_is_kwdarg = False
    self._name_is_posarg = False
    if 'name' in self._params:
      name_param = self._params['name']
      kind = name_param.kind
      if kind == inspect.Parameter.KEYWORD_ONLY:
        self._name_is_kwdarg = True
      if kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
        self._name_is_kwdarg = True
        self._name_is_posarg = True
      else:
        self._name_is_posarg = True

  def apply_attrs(self, visitor, attrs):
    return PrimitiveFunction(partial(self._fn, **attrs))

  def apply_kw(self, visitor, ctx, name, attrs, kwargs):
    if kwargs == None:
      kwargs = {}

    if name != None:
      kwargs = dict(kwargs)
      kwargs['name'] = name

    args = []
    if self._prepend_with_context:
      args.append(ctx)

    try:
      return self._fn(*args, **kwargs)
    except:
      raise Exception("Tried to call %s with args %s and kwargs %s"  % (self._fn, args, kwargs))

  def apply(self, visitor, ctx, name, attrs, args):
    if attrs == None:
      attrs = {}

    name_is_kwdarg = self._name_is_kwdarg

    if self._name_is_posarg:
      new_args = []
      arg_ix = 0
      nargs = len(args)
      for param_name, param in self._params.items():
        if param_name == 'name':
          new_args.append(name)
          name_is_kwdarg = False
        else:
          if arg_ix >= nargs:
            break
          new_args.append(args[arg_ix])
          arg_ix += 1
      args = new_args

    if name_is_kwdarg:
      attrs = dict(attrs)
      attrs['name'] = name

    if self._prepend_with_context:
      args = [ctx, *args]

    # eprint("Applying", "call %s with name %s args %s and kwargs %s"  % (self._fn, name, args, attrs))

    return ctx.call(self._fn, args, attrs)

# HACK(adamb) We need to be able to pass in our own custom instances of

from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops

class _TensorRefWrapper(optimizer._OptimizableVariable):
  def __init__(self, ref):
    self._ref = ref

  def target(self):
    return self._ref

  def update_op(self, optimizer, g):
    if isinstance(g, ops.Tensor):
      return optimizer._apply_dense(g, self._ref)  # pylint: disable=protected-access
    else:
      assert isinstance(g, ops.IndexedSlices), ("Gradient ", g, " is neither a "
                                                "tensor nor IndexedSlices.")
      # pylint: disable=protected-access
      return optimizer._apply_sparse_duplicate_indices(g, self._ref)

prev_get_processor = optimizer._get_processor
def _get_processor(v):
  if isinstance(v, tf.Tensor):
    return _TensorRefWrapper(v)
  return prev_get_processor(v)
optimizer._get_processor = _get_processor

class TransformedFunction:
  def __init__(self, name, fn, macro):
    self._nam = name
    self._fn = fn
    self._macro = macro

  def _name(self):
    return self._nam

  def _apply(self, impl, visitor):
    vars = []
    def var_reference(var):
      vars.append(var)

    eprint("about to apply!")
    visitor.add_variable_listener(var_reference)
    try:
      retvals = impl()
    finally:
      visitor.remove_variable_listener(var_reference)

    output = retvals.get(None)

    macro_attrs = {
      "output": output,
      "trainable": vars,
    }

    # eprint("macro_attrs", macro_attrs)

    out = self._macro.apply_attrs(visitor, macro_attrs)

    with tf.control_dependencies([out.get(None)]):
      return tf.identity(output)


  def apply_kw(self, visitor, ctx, scope_name, attrs, kwargs):
    def _apply_kw():
      return self._fn.apply_kw(visitor, ctx, scope_name, None, kwargs)

    return self._apply(_apply_kw, visitor)

  def apply(self, visitor, ctx, scope_name, attrs, args):
    def _apply():
      return self._fn.apply(visitor, ctx, scope_name, None, args)

    return self._apply(_apply, visitor)

class ImportedPythonFunction:
  def __init__(self, fn):
    self._fn = fn
    sig = inspect.signature(fn)
    self._Tout = sig.return_annotation
    self._argnames = sig.parameters.keys()

  def __call__(self, *args):
    return tf.py_func(
        func=self._fn,
        inp=args,
        Tout=self._Tout,
        stateful=True, # TODO
        name=None)

class DeclaredMacro:
  def __init__(self, ctx, expr):
    self._ctx = ctx
    self._expr = expr

  def _name(self):
    return self._expr[0]

  def _attr_specs(self):
    return self._expr[1];

  def _retval_specs(self):
    return self._expr[2]

  def _retval_argnames(self):
    return [name for (_, name) in self._retval_specs()]

  def _body(self):
    return self._expr[3:]

  def _do_macro_apply(self, visitor, ctx):
    returned = {}

    g = tf.get_default_graph()
    scope_name = g.unique_name(self._name() or "macro", False).split("/")[-1]

    with tf.variable_scope(scope_name):
      # Need to visit expressions
      visitor._visit_exprs(ctx, self._body())

    for retval_name, retval_argname in self._retval_specs():
      returned[retval_name] = ctx.get_local(retval_argname)

    eprint("returned", returned)

    return RetvalBag(returned)

  # Update given attrs and return a new function.
  # (add separate data structure for tracking pre-specified and now unoverrideable values).
  def apply_attrs(self, visitor, attrs):
    ctx = self._ctx.duplicate()

    has_ellipsis = False
    for name, value in attrs.items():
      if name == '_ellipsis':
        has_ellipsis = True
        continue
      ctx.define_attr(name, value)

    missing_attributes = []
    for name, _, _ in self._attr_specs():
      if name not in attrs and not ctx.has_attr(name):
        missing_attributes.append(name)

    if has_ellipsis:
      if len(missing_attributes) == 0:
        raise Exception("Saw ... given, but not missing attributes")

      return DeclaredMacro(ctx, self._expr)
    else:
      if len(missing_attributes) > 0:
        raise Exception("No ... given and missing attributes: %s" % missing_attributes)

      return self._do_macro_apply(visitor, ctx)


  # If we see syntax like: foo(a: ?, b: ?) then it's a partial application.
  # For these, bind the values we have return a new function where these values are unoverrideable.
  def apply_partial():
    pass


class DeclaredFunction:
  def __init__(self, ctx, expr):
    self._ctx = ctx
    self._expr = expr

  def clone(self):
    return DeclaredFunction(self._ctx, self._expr)

  def rename(self, name):
    self._expr[0] = name

  def _name(self):
    return self._expr[0]

  def _attr_specs(self):
    return self._expr[1];

  def _arg_specs(self):
    return self._expr[2]

  def _arg_names(self):
    return [name for (name, shape, dtype) in self._arg_specs()]

  def _retval_specs(self):
    return self._expr[3]

  def _retval_argnames(self):
    return [name for (_, name) in self._retval_specs()]

  def _body(self):
    return self._expr[4:]

  # Update given attrs and return a new function.
  # (add separate data structure for tracking pre-specified and now unoverrideable values).
  def apply_attrs(self, visitor, attrs):
    ctx = self._ctx.duplicate()

    has_ellipsis = False
    for name, value in attrs.items():
      if name == '_ellipsis':
        has_ellipsis = True
        continue
      ctx.define_attr(name, value)

    missing_attributes = []
    for name, _, _ in self._attr_specs():
      if name not in attrs and not ctx.has_attr(name):
        missing_attributes.append(name)

    if has_ellipsis:
      if len(missing_attributes) == 0:
        raise Exception("Saw ... given, but not missing attributes")
    else:
      if len(missing_attributes) > 0:
        raise Exception("No ... given and missing attributes: %s" % missing_attributes)

    return DeclaredFunction(ctx, self._expr)

  # If we see syntax like: foo(a: ?, b: ?) then it's a partial application.
  # For these, bind the values we have return a new function where these values are unoverrideable.
  def apply_partial():
    pass

  def _do_apply(self, visitor, ctx, scope_name, attrs, bind_args):
    returned = {}
    new_ctx = ctx.duplicate_for(self._ctx)
    if attrs != None:
      for n, v in attrs.items():
        new_ctx.define_attr(n, v)

    g = tf.get_default_graph()

    # preload locals with references to input operations
    bind_args(new_ctx)

    if scope_name:
      with tf.variable_scope(scope_name):
        # Need to visit expressions
        visitor._visit_exprs(new_ctx, self._body())
    else:
      visitor._visit_exprs(new_ctx, self._body())

    for retval_name, retval_argname in self._retval_specs():
      returned[retval_name] = new_ctx.get_local(retval_argname)

    result = RetvalBag(returned)
    # if name:
    #   # HACK(adamb) The tf.identity call below just demands that the result is a Tensor.
    #   if len(returned) == 1:
    #     result = tf.identity(result.get(None), name=name)

    return result

  def apply_kw(self, visitor, ctx, scope_name, attrs, kwargs):
    def bind_args_by_name(new_ctx):
      for arg_name, arg in kwargs.items():
        new_ctx.define_local(arg_name, arg)

    return self._do_apply(visitor, ctx, scope_name, attrs, bind_args_by_name)

  def apply(self, visitor, ctx, scope_name, attrs, args):
    def bind_args_by_pos(new_ctx):
      for arg_name, arg in zip(self._arg_names(), args):
        new_ctx.define_local(arg_name, arg)

    return self._do_apply(visitor, ctx, scope_name, attrs, bind_args_by_pos)
