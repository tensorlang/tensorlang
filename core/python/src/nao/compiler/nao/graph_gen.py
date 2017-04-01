import sys
import json

from functools import reduce
from functools import partial

import tensorflow as tf

from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import compat

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import control_flow_pb2

from nao.structure import graph_ffi
from nao.structure import graph_io

from nao.compiler.nao import graph_context
from nao.compiler.nao import graph_function
from nao.compiler.nao.graph_loop import _sf_while_loop

from nao.compiler.retvalbag import RetvalBag, unwrap_bag

from nao.compiler.python_package import PythonPackage

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)


class Nao:
  def __init__(self, visitor):
    self._visitor = visitor

  def map(self, ctx, elems, fn=None, dtype=None, name=None):
    eprint("map of", elems, "with", fn, dtype, "named", name)
    with tf.control_dependencies([]):
      try:
        def some_fn(elem):
          return fn.apply(self._visitor, ctx, None, None, [elem]).get(None)

        result = tf.map_fn(
          fn=some_fn,
          elems=elems,
          dtype=dtype,
          parallel_iterations=1,
          back_prop=False,
          swap_memory=False,
          infer_shape=True)

        return tf.identity(result, name=name)

      except KeyError as ke:
        eprint('error, but got graph', tf.get_default_graph().as_graph_def())
        nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        nodes.sort()
        eprint('error, but got nodes', nodes)
        raise ke

  def enqueue_many(self, ctx, queue_ref, components, name=None):
    if name is None:
      name = tf.get_default_graph().unique_name("EnqueueMany", False).split("/")[-1]

    ret = gen_data_flow_ops._queue_enqueue_many_v2(
        queue_ref, components=components, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Queue object.
    # op = ret[0].op
    # batch_dim = tensor_shape.Dimension(tensor_util.constant_value(op.inputs[1]))
    # for output, shape in zip(op.values(), shapes):
    #   output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))

    return ret

  def dequeue_many(self, ctx, queue_ref, n, component_types=None, name=None):
    if name is None:
      name = tf.get_default_graph().unique_name("DequeueMany", False).split("/")[-1]

    ret = gen_data_flow_ops._queue_dequeue_many_v2(
        queue_ref, n=n, component_types=component_types, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Queue object.
    # op = ret[0].op
    # batch_dim = tensor_shape.Dimension(tensor_util.constant_value(op.inputs[1]))
    # for output, shape in zip(op.values(), shapes):
    #   output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))

    return ret

  def dequeue(self, ctx, queue_ref, component_types=None, name=None):
    if name is None:
      name = tf.get_default_graph().unique_name("Dequeue", False).split("/")[-1]

    ret = gen_data_flow_ops._queue_dequeue_v2(
        queue_ref, component_types=component_types, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Queue object.
    # op = ret[0].op
    # batch_dim = tensor_shape.Dimension(tensor_util.constant_value(op.inputs[1]))
    # for output, shape in zip(op.values(), shapes):
    #   output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))

    return ret

  def var_transform(self, ctx, fn, macro, name=None):
    return graph_function.TransformedFunction(name, fn, macro)

class TopLevel:
  TYPES = {
	  "half": tf.float16,
	  "float": tf.float32,
	  "double": tf.float64,
	  "int8": tf.int8,
	  "int16": tf.int16,
	  "int32": tf.int32,
	  "int64": tf.int64,
	  "uint8": tf.uint8,
	  "uint16": tf.uint16,
	  "string": tf.string,
	  "bool": tf.bool,
	  "complex64": tf.complex64,
	  "complex128": tf.complex128,
	  "qint8": tf.qint8,
	  "qint32": tf.qint32,
	  "quint": tf.quint8,
  }

  def __init__(self):
    self.nesting_level = 0
    self._variable_listeners = []

  def add_variable_listener(self, listener):
    self._variable_listeners.append(listener)

  def remove_variable_listener(self, listener):
    self._variable_listeners.remove(listener)

  # "primitive" values
  def _sf_type(self, ctx, name):
    return TopLevel.TYPES[name]

  def shape(self, ctx, *dims):
    return tf.TensorShape(dims)

  def _sf_whole(self, ctx, digits):
    return int(digits)

  def _sf_fraction(self, ctx, decimal):
    return float(decimal)

  def _named_define_local(self, ctx, name, value):
    ctx.define_local(name, value)
    return value

  def _named_define_attr(self, ctx, name, value):
    ctx.define_attr(name, value)
    return value

  def _named_tensor(self, ctx, name, shape, dtype, value):
    op = None
    try:
      op = tf.constant(value, shape=shape, dtype=dtype, name=name)
    except TypeError as e:
      eprint("tf.constant(%s, shape=%s, dtype=%s, name=%s)" % (value, shape, dtype, name))
      raise e
    ctx.possible_leaf(op)

    if name != None:
      ctx.define_local(name, op)

    return op

  def _named_placeholder(self, ctx, name, shape, dtype):
    op = tf.placeholder(dtype, shape=shape, name=name)
    ctx.define_local(name, op)
    return op

  def __named_apply_prep(self, ctx, name, fn, attrs, do_apply):
    if attrs and '_ellipsis' in attrs:
      raise Exception("Can't use attribute ellipsis in function apply")

    fn = unwrap_bag(fn)

    base_scope_name = None
    if hasattr(fn, '_name'):
      base_scope_name = fn._name()

    if not base_scope_name:
      base_scope_name = 'fnc'

    g = tf.get_default_graph()
    scope_name = "q___%s" % g.unique_name(base_scope_name, False).split("/")[-1]

    if not hasattr(fn, 'apply') and hasattr(fn, 'apply_attrs'):
      fn = fn.apply_attrs(self, attrs)
      fn = unwrap_bag(fn)
      attrs = None

    result = do_apply(fn, scope_name)

    ctx.possible_leaf(result)

    if name != None:
      ctx.define_local(name, result)

      # HACK(adamb) The tf.identity call below just demands that the result is a Tensor.
      if isinstance(result, RetvalBag) and result.len() == 1:
        result = result.get(None)
      if isinstance(result, tf.Tensor):
        tf.identity(result, name=name)

    return result

  def _named_apply_keywords(self, ctx, name, fn, attrs, kwargs):
    def keyword_apply(unwrapped_fn, scope_name):
      unwrapped_kwargs = {}
      for key, value in kwargs.items():
        value = unwrap_bag(value)
        ctx.eliminate_leaf(value)
        unwrapped_kwargs[key] = value

      return unwrapped_fn.apply_kw(self, ctx, name, attrs, unwrapped_kwargs)
    return self.__named_apply_prep(ctx, name, fn, attrs, keyword_apply)

  def _named_apply(self, ctx, name, fn, attrs, *args):
    def positonal_apply(unwrapped_fn, scope_name):
      unwrapped_args = []
      for arg in args:
        arg = unwrap_bag(arg)
        ctx.eliminate_leaf(arg)
        unwrapped_args.append(arg)
      if hasattr(unwrapped_fn, 'apply'):
        return unwrapped_fn.apply(self, ctx, scope_name, attrs, unwrapped_args)
      else:
        raise Exception("Can't apply non-function %s with unwrapped args %s" % (unwrapped_fn, unwrapped_args))
    return self.__named_apply_prep(ctx, name, fn, attrs, positonal_apply)

  # TODO(adamb) Should take a name
  def _sf_cond(self, ctx, cond_expr, then_expr, else_expr):
    return tf.cond(
      pred=self.visit(ctx, cond_expr),
      fn1=lambda: self.visit(ctx.subcontext(), then_expr),
      fn2=lambda: self.visit(ctx.subcontext(), else_expr),
    )

  # TODO(adamb) Should take a list of targets. If targets corresponds to '*'
  # syntax, use leaves. Otherwise, use specific entries.
  def _sf_after_leaves(self, ctx, *exprs):
    leaves = ctx.leaves()
    eprint("_sf_after_leaves", leaves)
    with tf.control_dependencies(list(leaves)):
      return self._visit_exprs(ctx, exprs)

  def _visit_exprs(self, ctx, exprs):
    result = None
    for expr in exprs:
      result = self.visit(ctx, expr)
      ctx.possible_leaf(result)
      ctx.set_above(result)
    return result

  # TODO(adamb) Consider a "name" register, which specifies what the name of
  #     the *next* node defined should be. Upon definition, the name register
  #      should be cleared. This would simplify the current contracts.
  #
  # TODO(adamb) Should have name

  def _sf_while_loop(self, ctx, cond_expr, body_exprs, body_retvals, init_exprs):
    return _sf_while_loop(self, ctx, cond_expr, body_exprs, body_retvals, init_exprs)

  def _sf_local(self, ctx, name):
    # eprint(ctx)
    return ctx.get_local(name)

  def _sf_package_lookup(self, ctx, pkg_name):
    return ctx.imported_package(pkg_name)

  def list(self, ctx, *entries):
    return [unwrap_bag(e) for e in entries]

  def apply_attrs(self, ctx, function, attrs):
    return unwrap_bag(function).apply_attrs(self, attrs)

  def _sf_map(self, ctx, *entry_exprs):
    d = {}
    for name, value_expr in entry_exprs:
      if name in d:
        raise Exception("Already have keyword %s" % name)
      d[name] = self.visit(ctx, value_expr)
      # d[name] = unwrap_bag(self.visit(ctx, value_expr))
    return d

  def _sf_attr(self, ctx, name):
    # eprint(ctx)
    return ctx.get_attr(name)

  def _named_var_update(self, ctx, name, rhs):
    return ctx.update_local(name, rhs)

  def _named_var(self, ctx, name, shape, dtype, initializer):
    if initializer != None and not callable(initializer):
      # TODO(adamb) Check the shape of initializer
      shape = None

    v = tf.Variable(
        name=name,
        initial_value=initializer,
        expected_shape=shape,
        dtype=dtype)
    # eprint("named var", v, type(v))

    ctx.define_local(name, v)

  def assert_type(self, ctx, dtype, val):
    # TODO(adamb) Actually check type
    return val

  def assert_shape(self, ctx, shape, val):
    # TODO(adamb) Actually check shape
    # eprint('%s.set_shape(%s)' % (val, shape))
    val.set_shape(shape)
    # eprint('shape is now %s' % (val))
    return val

  def _sf_index(self, ctx, expr, index_expr):
    index = self.visit(ctx, index_expr)
    target = self.visit(ctx, expr)

    return ctx.get_index(target, index)

  def _sf_function(self, ctx, name, attr_spec_exprs, arg_spec_exprs, retval_specs, *body_expr):
    attr_specs = attr_spec_exprs and [(name, self._visit(ctx, shape), self._visit(ctx, type)) for (name, shape, type) in attr_spec_exprs]
    arg_specs = [(name, self._visit(ctx, shape), self._visit(ctx, type)) for (name, shape, type) in arg_spec_exprs]

    fn = graph_function.DeclaredFunction(ctx.subcontext(), [name, attr_specs, arg_specs, retval_specs, *body_expr])

    return fn

  def _sf_macro(self, ctx, name, *rest):
    return graph_function.DeclaredMacro(ctx.subcontext(), [name, *rest])

  def _sf_import(self, ctx, name_triples):
    # HACK(adamb) At the moment, assume that we're only talking about python)
    for name, import_path, tag in name_triples:
      pkg = None
      if import_path.startswith("tensorflow:"):
        package_path, scope = import_path.split(":", 1)
        py_module = tf
        if scope:
          parts = scope.split("/")
          py_module = reduce(lambda p, n: getattr(p, n), parts, py_module)
        pkg = PythonPackage(py_module)
      else:
        if "://" in import_path:
          import_path = import_path.split("://")[1]
        pkg = ctx.fully_qualified_package(import_path)

      ctx.import_package(name, pkg)

  # HACK(adamb) For now we manually export declared functions with initial capital letters.
  #     When functions are emitted as FunctionDefs, this can be removed.
  def _maybe_export_function(self, package_name, subctx, name, value):
    if not name[0].isupper():
      eprint("not capitalized", name)
      return

    value = unwrap_bag(value)
    eprint("considering", name)

    if not isinstance(value, graph_function.DeclaredFunction):
      eprint("isn't a declared function", type(value))
      return

    fn = value

    g = tf.get_default_graph()
    var_collection_name = "%s:variable_names" % (g.unique_name(name, False))
    var_set = set()
    def on_var(var):
      var_set.add(var.name)
    self.add_variable_listener(on_var)

    eprint("exporting", name, fn)
    with tf.variable_scope(name):
      with tf.variable_scope("inputs"):
        args = [tf.placeholder(arg_dtype, arg_shape, arg_name) for (arg_name, arg_shape, arg_dtype) in fn._arg_specs()]

      subctx2 = subctx.subcontext()
      fn.apply(self, subctx2, "_", None, args)

      with tf.variable_scope("outputs"):
        g = tf.get_default_graph()
        for (retval_name, retval_inner_name) in fn._retval_specs():
          tensor_prefix = "%s/%s" % (package_name, name)
          try:
            returned_tensor = g.get_tensor_by_name("%s/_/%s:0" % (tensor_prefix, retval_inner_name))
          except KeyError as ke:
            eprint("repeating lookup of %s in prefix %s for retval %s" % (retval_inner_name, tensor_prefix, retval_name))
            # If we fail to find the tensor above, perhaps it was just an input.
            try:
              returned_tensor = g.get_tensor_by_name("%s/inputs/%s:0" % (tensor_prefix, retval_inner_name))
            except KeyError:
              nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
              nodes.sort()
              eprint('error, but got nodes', nodes)
              raise ke

          tf.identity(returned_tensor, name=retval_name)

    for var_name in var_set:
      g.add_to_collection(var_collection_name, var_name)

    self.remove_variable_listener(on_var)

  def _sf_package(self, imports, pkg, name, *exprs):
    if pkg is None:
      superctx = graph_context.Context(graph_context.SentinelContextDelegate())
      superctx.import_package("tf", PythonPackage(tf))
      superctx.import_package("nao", PythonPackage(Nao(self), prepend_with_context=True))
      pkg = superctx.resolve_fully_qualified_package(name)

    ctx = pkg.ctx()
    prev_local_items = ctx.local_items()
    prev_attr_items = ctx.attr_items()

    for import_name, import_pkg in imports.items():
      ctx.define_fully_qualified_package(import_name, import_pkg)

    with tf.variable_scope(name):
      self._visit_exprs(ctx, exprs)

      for local_name, local_value in ctx.local_items():
        if (local_name, local_value) in prev_local_items:
          continue

        self._maybe_export_function(name, ctx, local_name, local_value)

      for attr_name, attr_value in ctx.attr_items():
        if (attr_name, attr_value) in prev_attr_items:
          continue

        self._maybe_export_function(name, ctx, attr_name, attr_value)

      eprint("%sctx: %s" % ('  ' * self.nesting_level, ctx))
      return pkg

  def visit(self, ctx, expr):
    return self._visit_result(self._visit(ctx, expr))

  def _visit_result(self, result):
    # TODO(adamb) What about retval bags that contain variables?
    is_tensor_ref = isinstance(result, tf.Tensor) and result.dtype._is_ref_dtype
    is_variable = isinstance(result, tf.Variable)
    if is_variable or is_tensor_ref:
      # if is_tensor_ref:
      #   eprint("saw tensor ref", result, result.op)
      # else:
      #   eprint("saw variable", result)

      for listener in self._variable_listeners:
        listener(result)

    return result

  def _visit(self, ctx, expr):
    self.nesting_level = self.nesting_level + 1
    eprint("%s%s" % ('  ' * self.nesting_level, expr))

    if type(expr) == list:
      expr_type = expr[0]
      if not isinstance(expr_type, str):
        raise Exception("Expression type isn't a string. Expression: %s" % expr)
      attr = getattr(self, expr_type)

      if expr_type.startswith("_sf_"): # Special form
        result = attr(ctx, *expr[1:])
      elif expr_type.startswith("_named_"): # name, then expressions
        eprint("%sctx: %s" % ('  ' * self.nesting_level, ctx))
        result = attr(ctx, expr[1], *[self.visit(ctx, subexpr) for subexpr in expr[2:]])
      else: # just expressions
        result = attr(ctx, *[self.visit(ctx, subexpr) for subexpr in expr[1:]])

      # eprint("visited %s expr %s => %s; ctx: %s" % (expr_type, expr, result, ctx))
      eprint("%s=> %s" % ('  ' * self.nesting_level, result))
      self.nesting_level = self.nesting_level - 1
      return result
    else:
      # eprint("visiting primitive %s ctx: %s" % (expr, ctx))
      self.nesting_level = self.nesting_level - 1
      return expr
