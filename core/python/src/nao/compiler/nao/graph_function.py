import inspect
import re
import sys

import tensorflow as tf
from tensorflow.contrib.graph_editor import make_view
import tensorflow.contrib.graph_editor.transform as transform

from nao.compiler.retvalbag import RetvalBag

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

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

    if name and isinstance(value, tf.Tensor):
      return tf.identity(value, name=name)

    return value

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

    # eprint("returned", returned)

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

  def has_attrs(self):
    attr_specs = self._attr_specs()
    return attr_specs is not None and len(attr_specs) > 0

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

    with tf.variable_scope(scope_name):
      # Need to visit expressions
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
