import copy
import sys

from functools import reduce

import tensorflow as tf
from tensorflow.python.framework import meta_graph

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class RetvalBag:
  def __init__(self, a_dict):
    self._d = {}

    for k, v in a_dict.items():
      if type(v) == RetvalBag:
        raise Exception("Can't put a RetvalBag into another. %s" % a_dict)
      self._d[k] = v

  def get(self, key):
    if key == None:
      l = len(self._d)
      if l == 0:
        raise Exception("Can't get default retval for an empty RetvalBag")
      if l > 1:
        raise Exception("Can't get default retval a RetvalBag with more than one entry: %s" % self._d)
      key = list(self._d.keys())[0]

    return self._d[key]

  def length(self):
    return len(self._d)


class PrimitiveFunction:
  def __init__(self, fn):
    self._fn = fn

  def apply(self, visitor, name, kwargs, args):
    if kwargs == None:
      kwargs = {}

    if name != None:
      kwargs = dict(kwargs)
      kwargs['name'] = name

    try:
      return self._fn(*args, **kwargs)
    except:
      raise Exception("Tried to call %s with args %s and kwargs %s"  % (self._fn, args, kwargs))

class SyntheticFunction:
  def __init__(self, argnames, retnames, graphdef):
    self._argnames = argnames
    self._retnames = retnames
    self._graphdef = graphdef

  def apply(self, visitor, scope_name, kwargs, args):
    retvals = tf.import_graph_def(
      self._graphdef,
      name=scope_name,
      input_map=dict(zip(self._argnames, args)),
      return_elements=self._retnames)

    return RetvalBag(dict(zip(self._retnames, retvals)))

class Nao:
  def reasm(self, argvars, retvals, name=None):
    a = [argvar.name for argvar in argvars]
    r = [retval.name for retval in retvals]
    graph = argvars[0].graph

    return SyntheticFunction(a, r, graph.as_graph_def())

  def disasm(self, fn, name=None):
    argvars, retvals = fn.disasm()
    return RetvalBag({"inputs": argvars, "outputs": retvals})

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

  def get(self, key):
    if key == "outputs":
      return list(self._gen_cached()[1]._d.values())

    if key == "inputs":
      return self._gen_cached()[0]

    raise "Unknown key for function %s" % key

  def disasm(self):
    with tf.Graph().as_default() as g:
      # TODO(adamb) Should actually have captured the environment where the function was defined.
      visitor = TopLevel()
      new_ctx = self._ctx.duplicate()

      arg_vars = []
      for (arg_name, shape_expr, dtype_expr) in self._arg_specs():
        arg_var = tf.placeholder(
          name=arg_name,
          dtype=visitor.visit(new_ctx, dtype_expr),
          shape=visitor.visit(new_ctx, shape_expr),
        )
        arg_vars.append(arg_var)
        new_ctx.define_local(arg_name, arg_var)

      for expr in self._body():
        visitor.visit(new_ctx, expr)

      retvals = [new_ctx.get_local(retval_argname) for retval_argname in self._retval_argnames()]
      return (arg_vars, retvals)


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

  # How to represent RemyCC in this system? Can we do this tables approach?

  def apply(self, visitor, scope_name, attrs, args):
    returned = {}
    new_ctx = self._ctx.duplicate()
    if attrs != None:
      for name, value in attrs.items():
        new_ctx.define_attr(name, value)

    with tf.variable_scope(scope_name):
      # preload locals with references to input operations
      for arg, arg_name in zip(args, self._arg_names()):
        new_ctx.define_local(arg_name, arg)

      # Need to visit expressions
      for expr in self._body():
        result = visitor.visit(new_ctx, expr)
        new_ctx.set_above(result)

    for retval_name, retval_argname in self._retval_specs():
      returned[retval_name] = new_ctx.get_local(retval_argname)

    # For now we only use the first retval
    return RetvalBag(returned)


class Context:
  def __init__(self, parent=None):
    self._parent = parent
    self._namespaces = {}
    self._attrs = {}
    self._locals = {}
    self._root_suffixes = {}
    self._leaves = set()
    self._above = None

  def subcontext(self):
    return Context(parent=self)

  def duplicate(self):
    ctx = copy.copy(self)
    ctx._attrs = copy.copy(ctx._attrs)
    ctx._locals = copy.copy(ctx._locals)
    return ctx

  def set_above(self, value):
    self._above = value

  def set_namespace(self, name, value):
    if name in self._namespaces:
      raise Exception("Namespace already defined: %s" % name)
    self._namespaces[name] = value

  def namespace_lookup(self, ns_name, key):
    if ns_name in self._namespaces:
      ns = self._namespaces[ns_name]
      return PrimitiveFunction(getattr(ns, key));

    if self._parent:
      return self._parent.namespace_lookup(ns_name, key)

  def define_local(self, name, value):
    if name in self._locals:
      raise Exception("Local already defined: %s" % name)
    self._locals[name] = value

  def has_attr(self, name):
    return name in self._attrs

  def define_attr(self, name, value):
    if name in self._attrs:
      raise Exception("Attribute already defined: %s" % name)

    if name in self._locals:
      raise Exception("Can't define attribute. Local exists with name: %s" % name)

    self._attrs[name] = value

  def get_local(self, name):
    if name == '^':
      return self._above

    if name in self._locals:
      return self._locals[name]

    if name in self._attrs:
      return self._attrs[name]

    if self._parent:
      return self._parent.get_local(name)

    raise Exception("No such local or function: %s. Have: %s" % (name, self._locals))

  def get_attr(self, name):
    if name in self._attrs:
      return self._attrs[name]

    raise Exception("No such attribute: %s. Have: %s" % (name, self._attrs))

  def possible_leaf(self, op):
    t = type(op)
    if t == tf.Tensor or t == tf.Operation:
      self._leaves.add(op)

  def eliminate_leaf(self, op):
    t = type(op)
    if t == tf.Tensor or t == tf.Operation:
      self._leaves.discard(op)

    if self._parent:
      return self._parent.eliminate_leaf(op)

  def leaves(self):
    l = frozenset(self._leaves)
    if self._parent:
      l = l | self._parent.leaves()
    return l

  # TODO(adamb) Properly nest names for parents.
  def unique_name(self, root):
    if not root in self._root_suffixes:
      self._root_suffixes[root] = -1

    suffix = self._root_suffixes[root]
    suffix = suffix + 1
    self._root_suffixes[root] = suffix

    return "%s_%s" % (root, suffix)

  def __str__(self):
    return "%s" % self._locals

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
    self._builtin_namespaces = {
      "tf": tf,
      "nao": Nao(),
    }

  # "primitive" values
  def _sf_type(self, ctx, name):
    return TopLevel.TYPES[name]

  def _sf_shape(self, ctx, dims):
    return tf.TensorShape(dims)

  def _sf_whole(self, ctx, digits):
    return int(digits)

  def _sf_fraction(self, ctx, decimal):
    return float(decimal)

  def _sf_list(self, ctx, *exprs):
    return [self.__unwrap_bag(self.visit(ctx, expr)) for expr in exprs]

  def _sf_define_local(self, ctx, name, value_expr):
    value = self.visit(ctx, value_expr)
    ctx.define_local(name, value)
    return value

  def __unwrap_bag(self, bag):
    if type(bag) == RetvalBag:
      return bag.get(None)
    return bag

  def _named_tensor(self, ctx, name, shape, dtype, value):
    op = tf.constant(value, shape=shape, dtype=dtype, name=name)
    ctx.possible_leaf(op)

    if name != None:
      ctx.define_local(name, op)

    return op

  def _named_placeholder(self, ctx, name, shape, dtype):
    op = tf.placeholder(dtype, shape=shape, name=name)
    ctx.define_local(name, op)
    return op

  def _named_apply(self, ctx, name, fn, attrs, *args):
    if attrs and '_ellipsis' in attrs:
      raise Exception("Can't use attribute ellipsis in function apply")

    fn = self.__unwrap_bag(fn)

    scope_name = name
    if scope_name == None and hasattr(fn, '_name'):
      scope_name = ctx.unique_name(fn._name())

    unwrapped_args = []
    for arg in args:
      arg = self.__unwrap_bag(arg)
      ctx.eliminate_leaf(arg)
      unwrapped_args.append(arg)

    result = fn.apply(self, scope_name, attrs, unwrapped_args)
    ctx.possible_leaf(result)

    if name != None:
      ctx.define_local(name, result)

    return result

  def _sf_cond(self, ctx, cond_expr, then_expr, else_expr):
    return tf.cond(
      pred=self.visit(ctx, cond_expr),
      fn1=lambda: self.visit(ctx.subcontext(), then_expr),
      fn2=lambda: self.visit(ctx.subcontext(), else_expr),
    )

  def _sf_while_loop(self, ctx, cond_expr, body_exprs, body_retvals, var_list_exprs):
    var_list = [self.visit(ctx, expr) for expr in var_list_exprs]
    var_names = [var.name.split("/")[-1].split(":")[0] for var in var_list]

    def cond(*a):
      ctx2 = ctx.subcontext()
      for name, val in zip(var_names, a):
        ctx2.define_local(name, val)
      return self.visit(ctx2, cond_expr)

    def body(*a):
      ctx2 = ctx.subcontext()
      for name, val in zip(var_names, a):
        ctx2.define_local(name, val)

      for expr in body_exprs:
        result = None
        if expr[0] == "__retval":
          name = expr[1]
          subexpr = expr[2]
          result = self.visit(ctx2, subexpr)
        else:
          result = self.visit(ctx2, expr)
        ctx2.set_above(result)

      body_retval_dict = dict(body_retvals)
      return [ctx2.get_local(body_retval_dict[var_name]) for var_name in var_names]

    results = tf.while_loop(
      cond=cond,
      body=body,
      loop_vars=var_list,
      parallel_iterations=1,
    )
    if type(results) != list:
      results = [results]

    return RetvalBag(dict(zip(var_names, results)))

  def _sf_local(self, ctx, name):
    # eprint(ctx)
    return ctx.get_local(name)

  def _sf_namespace_lookup(self, ctx, ns_name, fn_name):
    return ctx.namespace_lookup(ns_name, fn_name)

  def apply_attrs(self, ctx, function, attrs):
    return function.apply_attrs(self, attrs)

  def _sf_attr(self, ctx, name):
    # eprint(ctx)
    return ctx.get_attr(name)

  # generating graphs directly
  def visit_graph_exprs(self, ctx, retval_names, exprs):
    for expr in exprs:
      result = None
      if expr[0] == "__retval":
        name = expr[1]
        subexpr = expr[2]
        result = self.visit(ctx, subexpr)
        ctx.define_local(name, result)
        retval_names.append(name)
      elif expr[0] == "__sf_after_leaves":
        # TODO(adamb) Should actually nest local variables AND leaves
        after_exprs = expr[1:]
        leaves = ctx.leaves()
        with tf.control_dependencies(leaves):
          result = self.visit_graph_exprs(ctx, retval_names, after_exprs)
      else:
        result = self.visit(ctx, expr)

      ctx.set_above(result)

  def _sf_graph(self, ctx, name, *exprs):
    with tf.variable_scope(name):
      retval_names = []
      local_ops = ctx.subcontext()

      with tf.variable_scope("_"):
        self.visit_graph_exprs(local_ops, retval_names, exprs)

      for retval_name in retval_names:
        op = local_ops.get_local(retval_name)
        tf.identity(op, name=retval_name)

  def _sf_index(self, ctx, expr, index_expr):
    index = self.visit(ctx, index_expr)
    target = self.visit(ctx, expr)
    if type(target) == RetvalBag:
      return target.get(index)

    return target[index]

  def _sf_function(self, ctx, name, *rest):
    return DeclaredFunction(ctx.subcontext(), [name, *rest])

  def _sf_attrs_with_ellipsis(self, ctx, *attr_exprs):
    attrs = self._sf_attrs(ctx, *attr_exprs)
    attrs['_ellipsis'] = True
    return attrs

  def _sf_attrs(self, ctx, *attr_exprs):
    attrs = {}
    for name, value_expr in attr_exprs:
      attrs[name] = self.visit(ctx, value_expr)
    return attrs

  def _sf_import(self, ctx, name_pairs):
    # HACK(adamb) At the moment, assume that we're only talking about python)
    for name, package_fragments in name_pairs:
      ns = reduce(
        lambda p, n: getattr(p, n),
        package_fragments[1:],
        {"tf": tf}[package_fragments[0]]
      )

      ctx.set_namespace(name, ns)

  def visit(self, ctx, expr):
    self.nesting_level = self.nesting_level + 1
    # eprint("%s%s" % ('  ' * self.nesting_level, expr))

    if type(expr) == list:
      expr_type = expr[0]
      attr = getattr(self, expr_type)

      if expr_type.startswith("_sf_"): # Special form
        result = attr(ctx, *expr[1:])
      elif expr_type.startswith("_named_"): # name, then expressions
        result = attr(ctx, expr[1], *[self.visit(ctx, subexpr) for subexpr in expr[2:]])
      else: # just expressions
        result = attr(ctx, *[self.visit(ctx, subexpr) for subexpr in expr[1:]])

      # eprint("visited %s expr %s => %s; ctx: %s" % (expr_type, expr, result, ctx))
      self.nesting_level = self.nesting_level - 1
      return result
    else:
      # eprint("visiting primitive %s ctx: %s" % (expr, ctx))
      self.nesting_level = self.nesting_level - 1
      return expr

def meta_graph_def_from_exprs(exprs):
  with tf.Graph().as_default() as g:
    visitor = TopLevel()
    ctx = Context()
    for name, ns in visitor._builtin_namespaces.items():
      ctx.set_namespace(name, ns)

    for expr in exprs:
      visitor.visit(ctx, expr)

    return meta_graph.export_scoped_meta_graph()[0]
