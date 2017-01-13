import sys

import tensorflow as tf

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class RetvalBag:
  def __init__(self, a_dict):
    self._d = {}

    for k, v in a_dict.items():
      if type(v) == RetvalBag:
        raise "Can't put a RetvalBag into another. %s" % a_dict
      self._d[k] = v

  def get(self, key):
    if key == None:
      l = len(self._d)
      if l == 0:
        raise "Can't get default retval for an empty RetvalBag"
      if l > 1:
        raise "Can't get default retval a RetvalBag with more than one entry: %s" % self._d
      key = list(self._d.keys())[0]

    return self._d[key]

  def length(self):
    return len(self._d)

class Context:
  def __init__(self):
    self.locals = {}
    self.root_suffixes = {}
    self._leaves = set()

  def set_local(self, name, value):
    self.locals[name] = value

  def get_local(self, name):
    return self.locals[name]

  def possible_leaf(self, op):
    t = type(op)
    if t == tf.Tensor or t == tf.Operation:
      self._leaves.add(op)

  def eliminate_leaf(self, op):
    t = type(op)
    if t == tf.Tensor or t == tf.Operation:
      self._leaves.discard(op)

  def leaves(self):
    return frozenset(self._leaves)

  def unique_name(self, root):
    if not root in self.root_suffixes:
      self.root_suffixes[root] = -1

    suffix = self.root_suffixes[root]
    suffix = suffix + 1
    self.root_suffixes[root] = suffix

    return "%s_%s" % (root, suffix)

  def __str__(self):
    return "%s" % self.locals

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
    self.functions = {}

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
    return [self.visit(ctx, expr) for expr in exprs]

  def _named_tensor(self, ctx, name, shape, dtype, value):
    op = tf.constant(value, shape=shape, dtype=dtype, name=name)
    ctx.possible_leaf(op)

    if name != None:
      ctx.set_local(name, op)
    return op

  def _named_placeholder(self, ctx, name, shape, dtype):
    op = tf.placeholder(dtype, shape=shape, name=name)
    ctx.set_local(name, op)
    return op

  # applying a function
  def _sf_apply(self, ctx, name, ns_name, fn_name, attrs_expr, *arg_exprs):
    # attrs = self.visit(ctx, attrs_expr)
    args = []
    for expr in arg_exprs:
      arg = self.visit(ctx, expr)
      if type(arg) == RetvalBag:
        arg = arg.get(None)
      ctx.eliminate_leaf(arg)
      # eprint("arg %s -> %s" % (expr, arg))
      args.append(arg)

    if ns_name != None:
      # For now assume ns is tf if non-None.
      ns = tf
      # How to handle multiple return values?
    #   eprint("tf.%s(%s)" % (fn_name, args))
      result = getattr(ns, fn_name)(*args, name=name)
      ctx.possible_leaf(result)
      if name != None:
        ctx.set_local(name, result)

      return result

    else:
      function = self.functions[fn_name]
      scope_name = name
      if scope_name == None:
        scope_name = ctx.unique_name(fn_name)

      returned = {}
      new_ctx = Context()
      with tf.variable_scope(scope_name):
        arg_specs, retval_specs, *body = function[1:]

        # preload locals with references to input operations
        for arg, arg_spec in zip(args, arg_specs):
          arg_name = arg_spec[0]
          new_ctx.set_local(arg_name, arg)

        # Need to visit expressions
        for expr in body:
          self.visit(new_ctx, expr)

      for retval_name, retval_argname in retval_specs:
        returned[retval_name] = new_ctx.get_local(retval_argname)

      # For now we only use the first retval
      result = RetvalBag(returned)
      if name != None:
        ctx.possible_leaf(result)
        ctx.set_local(name, result)

      return result

  def _sf_local(self, ctx, name):
    # eprint(ctx)
    return ctx.get_local(name)

  # generating graphs directly
  def visit_graph_exprs(self, ctx, retval_names, exprs):
    for expr in exprs:
      if expr[0] == "__retval":
        name = expr[1]
        subexpr = expr[2]
        op = self.visit(ctx, subexpr)
        ctx.set_local(name, op)
        retval_names.append(name)
      elif expr[0] == "__sf_after_leaves":
        # TODO(adamb) Should actually nest local variables AND leaves
        after_exprs = expr[1:]
        leaves = ctx.leaves()
        with tf.control_dependencies(leaves):
          self.visit_graph_exprs(ctx, retval_names, after_exprs)
      else:
        self.visit(ctx, expr)

  def _sf_graph(self, ctx, name, *exprs):
    with tf.variable_scope(name):
      retval_names = []
      local_ops = Context()

      with tf.variable_scope("_"):
        self.visit_graph_exprs(local_ops, retval_names, exprs)

      for retval_name in retval_names:
        op = local_ops.get_local(retval_name)
        tf.identity(op, name=retval_name)

  def _sf_index(self, ctx, expr, index):
    target = self.visit(ctx, expr)
    return target.get(index)

  def _sf_def_function(self, ctx, name, *rest):
    self.functions[name] = [name, *rest]

  def _sf_attrs(self, ctx):
    return {}

  def visit(self, ctx, expr):
    self.nesting_level = self.nesting_level + 1
    if type(expr) == list:
      expr_type = expr[0]
      eprint("%s%s" % ('  ' * self.nesting_level, expr))
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

def graph_def_from_exprs(exprs):
  with tf.Graph().as_default() as g:
    visitor = TopLevel()
    for expr in exprs:
      visitor.visit(Context(), expr)

    return g.as_graph_def()
