import sys
import imp
import json

from functools import reduce
from functools import partial

import tensorflow as tf

from tensorflow.python.framework import meta_graph
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import compat

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import control_flow_pb2

import graph_ffi
import graph_function
import graph_context

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


  def dequeue_many(self, ctx, queue_ref, n, component_types=None, name=None):
    if name is None:
      name = tf.get_default_graph().unique_name("DequeueMany").split("/")[-1]

    ret = gen_data_flow_ops._queue_dequeue_many(
        queue_ref, n=n, component_types=component_types, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Queue object.
    # op = ret[0].op
    # batch_dim = tensor_shape.Dimension(tensor_util.constant_value(op.inputs[1]))
    # for output, shape in zip(op.values(), shapes):
    #   output.set_shape(tensor_shape.TensorShape([batch_dim]).concatenate(shape))

    return ret

  def reasm(self, ctx, argvars, retvals, var_list):
    a = [argvar.name for argvar in argvars]
    r = [retval.name for retval in retvals]
    v = [var.to_proto() for var in var_list]
    graph = argvars[0].graph

    eprint("ctx", ctx)
    eprint("Creating SyntheticFunction with argvars %s, retvals %s, vars %s, vardefs %s" % (argvars, retvals, var_list, v))
    meta_graph_def = tf.train.export_meta_graph(graph=graph)
    # nodes = [n.name for n in meta_graph_def.graph_def.node]
    # nodes.sort()
    # eprint('exporting nodes named', nodes)
    return SyntheticFunction("asdfasdfs", a, r, v, meta_graph_def)

  def disasm(self, ctx, fn, name=None):
    argvars, retvals, trainable = fn.disasm()
    return graph_function.RetvalBag({"name": name, "inputs": argvars, "outputs": retvals, "trainable": trainable})


  # TODO(adamb) Need to think about how to properly identify variables to target
  #     for optimization. Do we want *all* variables in a function? (Probably.)
  #     Do we instead want only the variables *explicitly* contained by a function?
  #     And not those referenced by it indirectly? Who knows what that even means?
  #     we should probably just include all variables that a function closes over.
  #     Is this trivial to compute? The full list of all variables closed over by
  #     a function? We can apply the function to placeholders and do something
  #     special whenever we encounter reference to a variable? Rather than proxy
  #     anything in a different context, we might instead export the entire
  #     primitive tree. Need to think about this though. Perhaps we want/need a
  #     way to export a function/closure to a meta_graph_def. If we could convert
  #     these back and forth, we'd have a lot of power.

  def transform(self, ctx, fn, macro, name=None):
    with tf.Graph().as_default() as g:
      # TODO(adamb) Should actually have captured the environment where the function was defined.
      visitor = TopLevel()
      proxyctx = graph_context.ProxyContext(ctx)
      new_ctx = proxyctx.duplicate_for(fn._ctx)

      arg_vars = []
      for (arg_name, shape_expr, dtype_expr) in fn._arg_specs():
        eprint("(arg_name %s, shape_expr %s, dtype_expr %s)" % (arg_name, shape_expr, dtype_expr))
        arg_var = tf.placeholder(
          name=arg_name,
          dtype=visitor.visit(proxyctx, dtype_expr),
          shape=visitor.visit(proxyctx, shape_expr),
        )
        arg_vars.append(arg_var)
        proxyctx.define_local(arg_name, arg_var)

      visitor._visit_exprs(proxyctx, fn._body())

      retvals = [proxyctx.get_local(retval_argname) for retval_argname in fn._retval_argnames()]
      trainable = tf.all_variables()

      macro_attrs = {
        "name": name or fn._name(),
        "inputs": arg_vars,
        "outputs": retvals,
        "trainable": trainable,
      }
      try:
        return macro.apply_attrs(visitor, macro_attrs)
      except Exception as e:
        eprint("Encountered problem, graph is", g.as_graph_def())
        raise e


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

  # "primitive" values
  def _sf_type(self, ctx, name):
    return TopLevel.TYPES[name]

  def shape(self, ctx, *dims):
    return tf.TensorShape(dims)

  def _sf_whole(self, ctx, digits):
    return int(digits)

  def _sf_fraction(self, ctx, decimal):
    return float(decimal)

  def __unwrap_bag(self, v):
    if type(v) == graph_function.RetvalBag:
      v = v.get(None)
    # if type(v) == tf.Variable:
    #   v = v.initialized_value()
    return v

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

    fn = self.__unwrap_bag(fn)

    scope_name = name
    if scope_name == None and hasattr(fn, '_name'):
      g = tf.get_default_graph()
      scope_name = g.unique_name(fn._name() or 'fnc').split("/")[-1]

    if not hasattr(fn, 'apply') and hasattr(fn, 'apply_attrs'):
      fn = fn.apply_attrs(self, attrs)
      fn = self.__unwrap_bag(fn)
      attrs = None

    result = do_apply(fn, scope_name)

    ctx.possible_leaf(result)

    if name != None:
      ctx.define_local(name, result)

    return result

  def _named_apply_keywords(self, ctx, name, fn, attrs, kwargs):
    def keyword_apply(unwrapped_fn, name):
      unwrapped_kwargs = {}
      for key, value in kwargs.items():
        value = self.__unwrap_bag(value)
        ctx.eliminate_leaf(value)
        unwrapped_kwargs[key] = value

      return unwrapped_fn.apply_kw(self, ctx, name, attrs, unwrapped_kwargs)
    return self.__named_apply_prep(ctx, name, fn, attrs, keyword_apply)

  def _named_apply(self, ctx, name, fn, attrs, *args):
    def positonal_apply(unwrapped_fn, name):
      unwrapped_args = []
      for arg in args:
        arg = self.__unwrap_bag(arg)
        ctx.eliminate_leaf(arg)
        unwrapped_args.append(arg)
      if hasattr(unwrapped_fn, 'apply'):
        return unwrapped_fn.apply(self, ctx, name, attrs, unwrapped_args)
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
    with tf.control_dependencies(leaves):
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

  def _sf_while_inner(self, ctx, exprs):
    with tf.Graph().as_default() as g:
      visitor = TopLevel()
      final_tensor = visitor._visit_exprs(ctx, exprs)

      # We do not want to include shapes, since inferred shapes will cause problems
      # for shape inference upon import and re-export.
      graph_def = g.as_graph_def()
      return (
        tf.train.export_meta_graph(graph=g, graph_def=graph_def),
        self.__unwrap_bag(final_tensor).name
      )

  def _sf_while_embed(self, import_scope, input_map, retval_names, meta_graph_def):
    g = tf.get_default_graph()

    try:
      with tf.name_scope(None):
        # with tf.control_dependencies(None):
          try:
            eprint('while embed', import_scope, input_map, retval_names)
            imported_vars = tf.train.import_meta_graph(
                meta_graph_def,
                import_scope=import_scope,
                input_map=input_map)
            eprint('imported_vars', import_scope, imported_vars)
          except ValueError as ve:
            # HACK(adamb) We don't want to error on unused input_map values.
            pass

      # eprint("have graph", import_scope, g.as_graph_def(add_shapes=True))

      return [g.get_tensor_by_name("%s/%s" % (import_scope, n)) for n in retval_names]
    except KeyError as ke:
      eprint('error, but got graph', tf.get_default_graph().as_graph_def())
      nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
      nodes.sort()
      eprint('error, but got nodes', nodes)
      raise ke

  def _while_prune(self, meta_graph_def, prune_names):
    graph_def = meta_graph_def.graph_def
    nodes = graph_def.node
    for ix in range(len(nodes) - 1, -1, -1):
      n = nodes[ix]
      if n.name in prune_names:
        del nodes[ix]
        eprint("n.name removed from graph_def!!", n.name)
      else:
        eprint("n.name not in proxy_names", n.name)


  def _while_fix_context_scope(self, meta_graph_def, import_scope):
    col_defs = meta_graph_def.collection_def

    if "while_context" not in col_defs:
      return

    wc_values = col_defs["while_context"].bytes_list.value
    for wcd_ix in range(0, len(wc_values)):
      while_context_bytes = wc_values[wcd_ix]
      while_context_def = control_flow_pb2.WhileContextDef()
      while_context_def.ParseFromString(while_context_bytes)
      values_def = while_context_def.values_def
      values = values_def.values
      # for v_ix in range(0, len(values)):
      #   values[v_ix] = import_scope + "/" + values[v_ix]

      external_values = values_def.external_values
      # for k in list(external_values.keys()):
      #   external_values[k] = import_scope + "/" + external_values[k]

      eprint("while_context_def", while_context_def, while_context_def.SerializeToString())
      wc_values[wcd_ix] = while_context_def.SerializeToString()


  def _while_fix_colocations(self, meta_graph_def, input_map):
    graph_def = meta_graph_def.graph_def
    for node in graph_def.node:
      # Rewrite the colocation attributes in the graph, since the
      # names of new ops may have changed.
      for key, value in node.attr.items():
        if key != '_class':
          continue

        class_values = value.list.s
        for ix in range(len(class_values) - 1, -1, -1):
          class_value = class_values[ix]
          if class_value.startswith(b'loc:@'):
            op_name = class_value[5:].decode()
            if not op_name in input_map:
              eprint("Skipping replacement of", op_name)
              continue
            # replacement_name = input_map[op_name].name
            # replacement = compat.as_bytes("loc:@" + replacement_name)
            # eprint("Replacing", class_value, "with", replacement)
            # class_values[ix] = replacement
            # HACK(adamb) It would be much, much better to just do the replacement
            #     commented out above, but we apparently can't replace a location
            #     with a value pointing to the existing graph. Strange.
            eprint("HACK(adamb) Removing", class_value)
            del class_values[ix]

  def _sf_while_loop(self, ctx, cond_expr, body_exprs, body_retvals, init_exprs):
    # Need to evaluate body_exprs first, looking for all variables that will be created
    # internally. Roll up into nested variable contexts. Unroll these contexts to be
    # passed as part of var_list. Within def body(*a), repackage these variables into
    # variable contexts. Use these contexts *instead of* creating variables directly.
    # So we want to be able to indicate whether or not contexts should be allowed to
    # create variables on the fly. Within a while cond/body, they should not. Otherwise,
    # the can (e.g. when compiling a graph { ... } expression)

    eprint('init_exprs', init_exprs)
    initial_value_ctx = ctx.subcontext()
    initial_tensor_list = None
    initial_local_names = None
    with tf.variable_scope('while/init'):
      initial_tensor_list = [self.__unwrap_bag(self.visit(initial_value_ctx, expr)) for expr in init_exprs]
      initial_local_names = [define[1] for define in init_exprs]

    local_name_by_tensor_name = dict(zip([t.name for t in initial_tensor_list], initial_local_names))

    # We need to put placeholders in scratch graph for all init values,
    # all upvals
    proxyctx = graph_context.ProxyContext(initial_value_ctx)

    g = tf.get_default_graph()

    # Ensure we have a placeholder for every initial value.
    for local_name in initial_local_names:
      proxyctx.get_local(local_name)

    proxyctx.clear_placeholder_op_cache()
    cond_ctx = proxyctx.subcontext()
    cond_meta_graph_def, cond_retval_name = self._sf_while_inner(cond_ctx, [cond_expr])

    # Don't let cached placeholders from cond_exprs infect our graph.
    # Expect their names/shapes/types to be reused, but not the placeholder
    # instances themselves.
    proxyctx.clear_placeholder_op_cache()
    body_ctx = proxyctx.subcontext()
    body_meta_graph_def, _ = self._sf_while_inner(body_ctx, body_exprs)

    input_map = proxyctx.input_map()
    input_keys = proxyctx.input_keys()
    input_values = proxyctx.input_values()
    # input_map_shapes = [v.get_shape() for v in input_values]
    # eprint("while input_map_shapes", input_map_shapes)
    eprint("while names to remove", proxyctx.proxy_names())

    # HACK(adamb) Don't actually import any nodes that are only proxies.
    #     This should probably be done automatically by the TF import
    #     logic, but empirically this is not the case.
    proxy_names = proxyctx.proxy_names()
    self._while_prune(cond_meta_graph_def, proxy_names)
    self._while_fix_colocations(cond_meta_graph_def, proxy_names)

    self._while_prune(body_meta_graph_def, proxy_names)
    self._while_fix_colocations(body_meta_graph_def, proxy_names)

    body_retval_dict = dict(body_retvals)
    body_retval_names = []
    next_value_ixs = []
    loop_vars = input_values

    ix = -1
    for t in loop_vars:
      ix += 1
      # if it's in initial_tensor_list, then look up its init_local_name
      # if we have a retval for this init_local_name, then use the inner_retval
      # otherwise pass through.
      if t.name in local_name_by_tensor_name:
        local_name = local_name_by_tensor_name[t.name]
        if local_name in body_retval_dict:
          eprint("while next vals", ix, t.get_shape(), t.name, local_name, body_retval_dict[local_name])
          body_retval_names.append("%s:0" % body_retval_dict[local_name])
          next_value_ixs.append(ix)
        else:
          eprint("while next vals skipped", ix, local_name)
      else:
        eprint("while next vals t.name", ix, t.name)

    eprint("while initial_local_names", initial_local_names)
    eprint("while initial_tensor_list", initial_tensor_list)
    eprint("while input_map", input_map)
    eprint("while body_retvals", body_retvals)
    eprint("while body_retval_dict", body_retval_dict)
    eprint("while body_retval_names", body_retval_names)
    eprint("while cond_retval_name", cond_retval_name)
    eprint("while local_name_by_tensor_name", local_name_by_tensor_name)

    def cond(*a):
      # We use a variable_scope because name_scope has a strange
      # only-sometimes-present trailing / that messes with everything.
      with tf.variable_scope('while/cond') as import_scope:
        pass
      cond_import_scope = import_scope.name

      self._while_fix_context_scope(cond_meta_graph_def, cond_import_scope)

      return self._sf_while_embed(
          cond_import_scope,
          dict(zip(input_keys, a)),
          [cond_retval_name],
          cond_meta_graph_def)[0]

    def body(*a):
      body_input_map = dict(zip(input_keys, a))
      eprint("while body", body_input_map)

      # We use a variable_scope because name_scope has a strange
      # only-sometimes-present trailing / that messes with everything.
      with tf.variable_scope('while/body') as import_scope:
        pass
      body_import_scope = import_scope.name

      self._while_fix_context_scope(body_meta_graph_def, body_import_scope)

      next_values = self._sf_while_embed(
          body_import_scope,
          body_input_map,
          body_retval_names,
          body_meta_graph_def)

      body_results = list(a)
      eprint('while body a', a)
      eprint('while body_retval_names', body_retval_names)
      eprint('while next_values', next_values)
      for ix, val in zip(next_value_ixs, next_values):
        val.set_shape(a[ix].get_shape())
        # eprint('while shape', ix, a[ix], a[ix].get_shape(), val, val.get_shape())
        # val.set_shape(val.get_shape())
        body_results[ix] = val

      eprint('while body_results', body_results)
      return body_results

    eprint("tf.while_loop(cond=%s, body=%s, loop_vars=%s)" % (cond, body, loop_vars))

    results = None
    try:
      results = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars,
        parallel_iterations=1,
        back_prop=False,
      )
      # eprint("have graph", tf.get_default_graph().as_graph_def(add_shapes=True))
    except KeyError as ke:
      # eprint("error, but body_meta_graph_def is", body_meta_graph_def)
      # eprint("error, but cond_meta_graph_def is", cond_meta_graph_def)
      raise ke
    except ValueError as ve:
      # eprint("error, but body_meta_graph_def is", body_meta_graph_def)
      # eprint("error, but cond_meta_graph_def is", cond_meta_graph_def)
      raise ve

    if type(results) != list:
      results = [results]

    r = {}
    for k, v in zip(input_values, results):
      if k.name in local_name_by_tensor_name:
        r[local_name_by_tensor_name[k.name]] = v

    return graph_function.RetvalBag(r)

  def _sf_local(self, ctx, name):
    # eprint(ctx)
    return ctx.get_local(name)

  def _sf_package_lookup(self, ctx, pkg_name):
    return ctx.imported_package(pkg_name)

  def list(self, ctx, *entries):
    return [self.__unwrap_bag(e) for e in entries]

  def apply_attrs(self, ctx, function, attrs):
    return self.__unwrap_bag(function).apply_attrs(self, attrs)

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
      elif expr[0] == "_sf_after_leaves":
        # TODO(adamb) Should actually nest local variables AND leaves
        after_exprs = expr[1:]
        leaves = ctx.leaves()
        with tf.control_dependencies(leaves):
          result = self.visit_graph_exprs(ctx, retval_names, after_exprs)
      else:
        result = self.visit(ctx, expr)

      ctx.set_above(result)

  def _named_var(self, ctx, name, shape, dtype, initializer):
    if initializer != None and not callable(initializer):
      # TODO(adamb) Check the shape of initializer
      shape = None

    v = tf.get_variable(
        name=name,
        initializer=initializer,
        shape=shape,
        dtype=dtype)

    ctx.define_local(name, v)

  def _sf_graph(self, ctx, name, *exprs):
    with tf.variable_scope(name):
      retval_names = []
      ctx2 = ctx.subcontext()

      with tf.variable_scope("_"):
        self.visit_graph_exprs(ctx2, retval_names, exprs)

      for retval_name in retval_names:
        op = ctx2.get_local(retval_name)
        tf.identity(op, name=retval_name)

  def assert_type(self, ctx, dtype, val):
    # TODO(adamb) Actually check type
    return val

  def assert_shape(self, ctx, shape, val):
    # TODO(adamb) Actually check shape
    eprint('%s.set_shape(%s)' % (val, shape))
    val.set_shape(shape)
    eprint('shape is now %s' % (val))
    return val

  def _sf_index(self, ctx, expr, index_expr):
    index = self.visit(ctx, index_expr)
    target = self.visit(ctx, expr)

    return ctx.get_index(target, index)

  def _sf_function(self, ctx, name, *rest):
    return graph_function.DeclaredFunction(ctx.subcontext(), [name, *rest])

  def _sf_macro(self, ctx, name, *rest):
    return graph_function.DeclaredMacro(ctx.subcontext(), [name, *rest])

  def _sf_python_package(self, ctx, name, source):
    outer_module = imp.new_module("%s$wrapper" % name)
    all_fns = self._python_importer.import_module(name, source)
    for fn_name, fn in all_fns.items():
      imported_py_func = graph_function.ImportedPythonFunction(fn)
      setattr(outer_module, fn_name, imported_py_func)

    pkg = graph_function.PythonPackage(outer_module)
    ctx.define_fully_qualified_package(name, pkg)
    return pkg

  def _sf_import(self, ctx, name_pairs):
    # HACK(adamb) At the moment, assume that we're only talking about python)
    for name, package_path in name_pairs:
      pkg = None
      if package_path.startswith("tensorflow:"):
        py_module = tf
        suffix = package_path.split(":", 2)[1]
        if suffix:
          parts = suffix.split("/")
          py_module = reduce(lambda p, n: getattr(p, n), parts, py_module)
        pkg = graph_function.PythonPackage(py_module)
      else:
        # TODO(adamb) Stop doing splitting in parser. Split above in python-specific code.
        pkg = ctx.fully_qualified_package(package_path)

      ctx.import_package(name, pkg)

  def _sf_package(self, ctx, name, *exprs):
    subctx = ctx.subcontext()
    with tf.variable_scope(name):
      for expr in exprs:
        self.visit(subctx, expr)

      pkg = graph_function.Package(subctx)
      ctx.define_fully_qualified_package(name, pkg)
      return pkg

  def visit(self, ctx, expr):
    self.nesting_level = self.nesting_level + 1
    # eprint("%s%s" % ('  ' * self.nesting_level, expr))

    if type(expr) == list:
      expr_type = expr[0]
      if not isinstance(expr_type, str):
        raise Exception("Expression type isn't a string. Expression: %s" % expr)
      attr = getattr(self, expr_type)

      if expr_type.startswith("_sf_"): # Special form
        result = attr(ctx, *expr[1:])
      elif expr_type.startswith("_named_"): # name, then expressions
        result = attr(ctx, expr[1], *[self.visit(ctx, subexpr) for subexpr in expr[2:]])
      else: # just expressions
        result = attr(ctx, *[self.visit(ctx, subexpr) for subexpr in expr[1:]])

      # eprint("visited %s expr %s => %s; ctx: %s" % (expr_type, expr, result, ctx))
      # eprint("%s%s" % ('  ' * self.nesting_level, result))
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
    ctx = graph_context.Context(graph_context.SentinelContextDelegate())
    ctx.import_package("tf", graph_function.PythonPackage(tf))
    ctx.import_package("nao", graph_function.PythonPackage(Nao(visitor), prepend_with_context=True))

    visitor._visit_exprs(ctx, exprs)

    # NOTE(adamb) Could also store files to copy out in assets_collection
    py_func_data = visitor._python_importer.dump_py_funcs(script_ops._py_funcs)
    # eprint('saved py_func_data', py_func_data)

    js_py_func_data = tf.constant(json.dumps(py_func_data), name="py_funcs_json")

    meta_graph_def, _ = meta_graph.export_scoped_meta_graph()
    return meta_graph_def
