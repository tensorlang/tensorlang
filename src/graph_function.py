import inspect
import sys

import tensorflow as tf

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

  def apply(self, visitor, ctx, name, attrs, args):
    n, *_ = args
    value = self._ctx.get_local_strict(n)

    if not n[0].isupper():
      raise Exception("Tried to use non-exported value named: %s" % n)

    return value

class PythonPackage:
  def __init__(self, mod, prepend_with_context=False):
    self._mod = mod
    self._prepend_with_context = prepend_with_context

  def apply(self, visitor, ctx, name, attrs, args):
    eprint("applying %s to args %s" % (self, args))
    n, *_ = args
    if n.startswith('__'):
      raise Exception("Tried to use non-exported namespace entry named: %s" % n)

    val = getattr(self._mod, n)
    if callable(val):
      return PrimitiveFunction(val, self._prepend_with_context)

    return val

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

    eprint("Applying", "call %s with name %s args %s and kwargs %s"  % (self._fn, name, args, attrs))

    return ctx.call(self._fn, args, attrs)

class SyntheticFunction:
  def __init__(self, name, argnames, retnames, base_input_map, metagraphdef):
    self._nam = name
    self._argnames = argnames
    self._retnames = retnames
    self._base_input_map = base_input_map
    # self._var_scope = None
    # self._vardefs = vardefs
    self._metagraphdef = metagraphdef
    # self._did_import_vars = False
    # self._imported_vars = None
    # self._var_init = None

  # def _lazy_var_init(self):
  #   self._lazy_define_vars()
  #   return self._var_init
  #
  # def _lazy_define_vars(self):
  #   if not self._did_import_vars:
  #     self._imported_vars = self._do_import_vars()
  #     self._var_init = tf.group(self._imported_vars.values)
  #     self._did_import_vars = True
  #
  #   return self._imported_vars
  #
  # def _do_import_vars(self):
  #   m = {}
  #   for vardef in self._vardefs:
  #     eprint("defining var", vardef)
  #     m[vardef.name] = tf.Variable(variable_def=vardef, import_scope=self._var_scope)
  #
  #   return m

  def _name(self):
    return self._nam

  def _input_map(self, existing):
    existing = dict(existing)
    existing.update(self._base_input_map)
    return existing
    # varmap = self._lazy_define_vars()
    #
    # if len(varmap) == 0:
    #   return existing
    #
    # existing = dict(existing)
    # existing.update(varmap)
    # return existing

  def _do_apply(self, scope_name, arg_dict):
    # scope = tf.variable_scope(scope_name)
    # scope_name = tf.get_default_graph().unique_name(scope_name)
    scope_name = "main/testMnistFormat/_/while/retval0"
    # with tf.variable_scope(root_scope):
    g = tf.get_default_graph()
    with tf.name_scope(None):
      # with tf.name_scope("synth") as scope:
      eprint('_do_apply', scope_name, arg_dict)
      var_list = None
      try:
        var_list = tf.train.import_meta_graph(
          self._metagraphdef,
          import_scope=scope_name,
          input_map=arg_dict)
      except KeyError as e:
        nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        nodes.sort()
        eprint('error, but got nodes', nodes)
        raise e
      except ValueError as ve:
        eprint('error, but tried to import', self._metagraphdef)
        raise ve
      # eprint('apply_kw var_list', var_list)

    retvals = [g.get_tensor_by_name("%s/%s" % (scope_name, n)) for n in self._retnames]

    return RetvalBag(dict(zip(self._retnames, retvals)))

  def apply_kw(self, visitor, ctx, scope_name, attrs, kwargs):
    return self._do_apply(scope_name, self._input_map(kwargs))

  # TODO(adamb) Support attrs for synthetic functions!
  def apply(self, visitor, ctx, scope_name, attrs, args):
    return self._do_apply(scope_name, self._input_map(dict(zip(self._argnames, args))))


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

    eprint("macro_attrs", macro_attrs)

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
    scope_name = g.unique_name(self._name() or "macro").split("/")[-1]

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

  def _do_apply(self, visitor, ctx, name, attrs, bind_args):
    returned = {}
    new_ctx = ctx.duplicate_for(self._ctx)
    if attrs != None:
      for name, value in attrs.items():
        new_ctx.define_attr(name, value)

    g = tf.get_default_graph()
    scope_name = g.unique_name(self._name() or 'func').split("/")[-1]
    with tf.variable_scope(scope_name):
      # preload locals with references to input operations
      bind_args(new_ctx)

      # Need to visit expressions
      visitor._visit_exprs(new_ctx, self._body())

    for retval_name, retval_argname in self._retval_specs():
      returned[retval_name] = new_ctx.get_local(retval_argname)

    rb = RetvalBag(returned)
    # HACK(adamb) The tf.identity call below just demands that the result is a Tensor.
    if name and len(returned) == 1:
      tf.identity(rb.get(None), name=name)

    return rb

  def apply_kw(self, visitor, ctx, name, attrs, kwargs):
    def bind_args_by_name(new_ctx):
      for arg_name, arg in kwargs.items():
        new_ctx.define_local(arg_name, arg)

    return self._do_apply(visitor, ctx, name, attrs, bind_args_by_name)

  def apply(self, visitor, ctx, name, attrs, args):
    def bind_args_by_pos(new_ctx):
      for arg_name, arg in zip(self._arg_names(), args):
        new_ctx.define_local(arg_name, arg)

    return self._do_apply(visitor, ctx, name, attrs, bind_args_by_pos)
