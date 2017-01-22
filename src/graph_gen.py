import copy
import sys
import imp
import inspect
import json

from functools import reduce

import tensorflow as tf

from tensorflow.python.framework import meta_graph

import graph_ffi

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
      key = self._default_key()
    return self._d[key]

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

  def apply(self, visitor, name, attrs, args):
    n, *_ = args
    value = self._ctx.get_local_strict(n)

    if not n[0].isupper():
      raise Exception("Tried to use non-exported value named: %s" % n)

    return value

class PythonPackage:
  def __init__(self, mod):
    self._mod = mod

  def apply(self, visitor, name, attrs, args):
    eprint("applying %s to args %s" % (self, args))
    n, *_ = args
    if n.startswith('__'):
      raise Exception("Tried to use non-exported namespace entry named: %s" % n)

    return PrimitiveFunction(getattr(self._mod, n))

class PrimitiveFunction:
  def __init__(self, fn):
    self._fn = fn

  def apply_kw(self, visitor, name, attrs, kwargs):
    if kwargs == None:
      kwargs = {}

    if name != None:
      kwargs = dict(kwargs)
      kwargs['name'] = name

    try:
      return self._fn(**kwargs)
    except:
      raise Exception("Tried to call %s with args %s and kwargs %s"  % (self._fn, args, kwargs))

  def apply(self, visitor, name, attrs, args):
    if attrs == None:
      attrs = {}

    if name != None:
      attrs = dict(attrs)
      attrs['name'] = name

    try:
      return self._fn(*args, **attrs)
    except:
      raise Exception("Tried to call %s with args %s and attrs %s"  % (self._fn, args, attrs))

class SyntheticFunction:
  def __init__(self, argnames, retnames, graphdef):
    self._argnames = argnames
    self._retnames = retnames
    self._graphdef = graphdef

  def apply_kw(self, visitor, name, attrs, kwargs):
    retvals = tf.import_graph_def(
      self._graphdef,
      name=scope_name,
      input_map=kwargs,
      return_elements=self._retnames)

    return RetvalBag(dict(zip(self._retnames, retvals)))

  # TODO(adamb) Support attrs for synthetic functions!
  def apply(self, visitor, scope_name, attrs, args):
    retvals = tf.import_graph_def(
      self._graphdef,
      name=scope_name,
      input_map=dict(zip(self._argnames, args)),
      return_elements=self._retnames)

    return RetvalBag(dict(zip(self._retnames, retvals)))

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

  def _do_apply(self, visitor, scope_name, attrs, bind_args):
    returned = {}
    new_ctx = self._ctx.duplicate()
    if attrs != None:
      for name, value in attrs.items():
        new_ctx.define_attr(name, value)

    with tf.variable_scope(scope_name):
      # preload locals with references to input operations
      bind_args(new_ctx)

      # Need to visit expressions
      for expr in self._body():
        result = visitor.visit(new_ctx, expr)
        new_ctx.set_above(result)

    for retval_name, retval_argname in self._retval_specs():
      returned[retval_name] = new_ctx.get_local(retval_argname)

    return RetvalBag(returned)

  def apply_kw(self, visitor, scope_name, attrs, kwargs):
    def bind_args_by_name(new_ctx):
      for arg_name, arg in kwargs.items():
        new_ctx.define_local(arg_name, arg)

    return self._do_apply(visitor, scope_name, attrs, bind_args_by_name)

  def apply(self, visitor, scope_name, attrs, args):
    def bind_args_by_pos(new_ctx):
      for arg_name, arg in zip(self._arg_names(), args):
        new_ctx.define_local(arg_name, arg)

    return self._do_apply(visitor, scope_name, attrs, bind_args_by_pos)


class Context:
  def __init__(self, parent=None):
    self._parent = parent
    self._fully_qualified_packages = {}
    self._imported_packages = {}
    self._attrs = {}
    self._locals = {}
    self._root_suffixes = {}
    self._leaves = set()
    self._above = None

  def subcontext(self):
    return Context(parent=self)

  def define_fully_qualified_package(self, name, pkg):
    if name in self._fully_qualified_packages:
      raise Exception("Already defined package: %s" % name)

    self._fully_qualified_packages[name] = pkg

  def fully_qualified_package(self, name):
    if name in self._fully_qualified_packages:
      return self._fully_qualified_packages[name]

    if self._parent:
      return self._parent.fully_qualified_package(name)

    raise Exception("Package not available: %s" % name)

  def import_package(self, name, pkg):
    if name in self._imported_packages:
      raise Exception("Already imported package: %s" % name)

    self._imported_packages[name] = pkg

  def imported_package(self, name):
    if name in self._imported_packages:
      return self._imported_packages[name]

    if self._parent:
      return self._parent.imported_package(name)

    raise Exception("Package not imported: %s" % name)

  def duplicate(self):
    ctx = copy.copy(self)
    ctx._attrs = copy.copy(ctx._attrs)
    ctx._locals = copy.copy(ctx._locals)
    return ctx

  def set_above(self, value):
    self._above = value

  def define_local(self, name, value):
    if name in self._locals:
      raise Exception("Local already defined: %s" % name)
    self._locals[name] = value

  def has_attr(self, name):
    return name in self._attrs

  def define_attr(self, name, value):
    if self.has_attr(name):
      raise Exception("Attribute already defined: %s" % name)

    if name in self._locals:
      raise Exception("Can't define attribute. Local exists with name: %s" % name)

    self._attrs[name] = value

  def get_attr(self, name):
    if name in self._attrs:
      return self._attrs[name]

    raise Exception("No such attribute: %s. Have: %s" % (name, self._attrs))

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

  def get_local_strict(self, name):
    if name in self._locals:
      return self._locals[name]

    raise Exception("No such entry: %s. Have: %s" % (name, self._locals))

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
    self._python_importer = graph_ffi.PythonImporter()
    self._builtin_namespaces = {
      "tf": tf,
      "py": Py(),
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

  def __named_apply_prep(self, ctx, name, fn, attrs, do_apply):
    if attrs and '_ellipsis' in attrs:
      raise Exception("Can't use attribute ellipsis in function apply")

    fn = self.__unwrap_bag(fn)

    scope_name = name
    if scope_name == None and hasattr(fn, '_name'):
      scope_name = ctx.unique_name(fn._name())

    result = do_apply(fn, scope_name)

    ctx.possible_leaf(result)

    if name != None:
      ctx.define_local(name, result)

    return result

  def _named_apply_keywords(self, ctx, name, fn, attrs, kwargs):
    def keyword_apply(unwrapped_fn, scope_name):
      unwrapped_kwargs = {}
      for key, value in kwargs.items():
        value = self.__unwrap_bag(value)
        ctx.eliminate_leaf(value)
        unwrapped_kwargs[key] = value

      return fn.apply_kw(self, scope_name, attrs, unwrapped_kwargs)
    return self.__named_apply_prep(ctx, name, fn, attrs, keyword_apply)

  def _named_apply(self, ctx, name, fn, attrs, *args):
    def positonal_apply(unwrapped_fn, scope_name):
      unwrapped_args = []
      for arg in args:
        arg = self.__unwrap_bag(arg)
        ctx.eliminate_leaf(arg)
        unwrapped_args.append(arg)
      return fn.apply(self, scope_name, attrs, unwrapped_args)
    return self.__named_apply_prep(ctx, name, fn, attrs, positonal_apply)

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

  def _sf_package_lookup(self, ctx, pkg_name):
    return ctx.imported_package(pkg_name)

  def list(self, ctx, *entries):
    return [self.__unwrap_bag(e) for e in entries]

  def apply_attrs(self, ctx, function, attrs):
    return function.apply_attrs(self, attrs)

  def _sf_map(self, ctx, *entry_exprs):
    d = {}
    for name, value_expr in entry_exprs:
      if name in d:
        raise Exception("Already have keyword %s" % name)
      d[name] = self.visit(ctx, value_expr)
      # d[name] = self.__unwrap_bag(self.visit(ctx, value_expr))
    return d

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

  def _sf_python_package(self, ctx, name, source):
    outer_module = imp.new_module("%s$wrapper" % name)
    all_fns = self._python_importer.import_module(name, source)
    for fn_name, fn in all_fns.items():
      imported_py_func = ImportedPythonFunction(fn)
      setattr(outer_module, fn_name, imported_py_func)

    pkg = PythonPackage(outer_module)
    ctx.define_fully_qualified_package(name, pkg)
    return pkg

  def _sf_import(self, ctx, name_pairs):
    # HACK(adamb) At the moment, assume that we're only talking about python)
    for name, package_fragments in name_pairs:
      pkg = None
      if package_fragments[0] == "tf":
        pkg = PythonPackage(
            reduce(
                lambda p, n: getattr(p, n),
                package_fragments[1:],
                tf))
      else:
        # TODO(adamb) Stop doing splitting in parser. Split above in python-specific code.
        pkg = ctx.fully_qualified_package(str.join("/", package_fragments))

      ctx.import_package(name, pkg)

  def _sf_package(self, ctx, name, *exprs):
    subctx = ctx.subcontext()
    with tf.variable_scope(name):
      for expr in exprs:
        self.visit(subctx, expr)

      pkg = Package(subctx)
      ctx.define_fully_qualified_package(name, pkg)
      return pkg

  def visit(self, ctx, expr):
    self.nesting_level = self.nesting_level + 1
    eprint("%s%s" % ('  ' * self.nesting_level, expr))

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

from tensorflow.python.ops import script_ops

def meta_graph_def_from_exprs(exprs):
  with tf.Graph().as_default() as g:
    visitor = TopLevel()
    ctx = Context()
    ctx.import_package("tf", PythonPackage(tf))

    for expr in exprs:
      visitor.visit(ctx, expr)

    # NOTE(adamb) Could also store files to copy out in assets_collection
    py_func_data = visitor._python_importer.dump_py_funcs(script_ops._py_funcs)
    # eprint('saved py_func_data', py_func_data)

    js_py_func_data = tf.constant(json.dumps(py_func_data), name="py_funcs_json")

    meta_graph_def, _ = meta_graph.export_scoped_meta_graph()
    return meta_graph_def
