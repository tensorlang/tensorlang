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
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import control_flow_pb2

from nao import graph_context
from nao import graph_ffi
from nao import graph_function
from nao import graph_io

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def _while_prune(meta_graph_def, prune_names):
  graph_def = meta_graph_def.graph_def
  nodes = graph_def.node
  for ix in range(len(nodes) - 1, -1, -1):
    n = nodes[ix]
    if n.name in prune_names:
      del nodes[ix]
      # eprint("n.name removed from graph_def!!", n.name)
    else:
      # eprint("n.name not in proxy_names", n.name)
      pass


def _while_fix_context_scope(meta_graph_def, import_scope):
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

    # eprint("while_context_def", while_context_def, while_context_def.SerializeToString())
    wc_values[wcd_ix] = while_context_def.SerializeToString()


def _while_fix_colocations(meta_graph_def, input_map):
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
            # eprint("Skipping replacement of", op_name)
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
    self._python_importer = graph_ffi.PythonImporter()
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
      scope_name = g.unique_name(fn._name() or 'fnc', False).split("/")[-1]

    if not hasattr(fn, 'apply') and hasattr(fn, 'apply_attrs'):
      fn = fn.apply_attrs(self, attrs)
      fn = self.__unwrap_bag(fn)
      attrs = None

    result = do_apply(fn, scope_name)

    ctx.possible_leaf(result)

    if name != None:
      ctx.define_local(name, result)

      # # HACK(adamb) The tf.identity call below just demands that the result is a Tensor.
      # if len(result) == 1:
      #   result = tf.identity(result.get(None), name=name)
      #
      # ctx.define_local(name, result)

    return result

  def _named_apply_keywords(self, ctx, name, fn, attrs, kwargs):
    def keyword_apply(unwrapped_fn, scope_name):
      unwrapped_kwargs = {}
      for key, value in kwargs.items():
        value = self.__unwrap_bag(value)
        ctx.eliminate_leaf(value)
        unwrapped_kwargs[key] = value

      return unwrapped_fn.apply_kw(self, ctx, name, attrs, unwrapped_kwargs)
    return self.__named_apply_prep(ctx, name, fn, attrs, keyword_apply)

  def _named_apply(self, ctx, name, fn, attrs, *args):
    def positonal_apply(unwrapped_fn, scope_name):
      unwrapped_args = []
      for arg in args:
        arg = self.__unwrap_bag(arg)
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
            # eprint('while embed', import_scope, input_map, retval_names)
            imported_vars = tf.train.import_meta_graph(
                meta_graph_def,
                import_scope=import_scope,
                input_map=input_map)
            # eprint('imported_vars', import_scope, imported_vars)
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

  def _sf_while_loop(self, ctx, cond_expr, body_exprs, body_retvals, init_exprs):
    # Need to evaluate body_exprs first, looking for all variables that will be created
    # internally. Roll up into nested variable contexts. Unroll these contexts to be
    # passed as part of var_list. Within def body(*a), repackage these variables into
    # variable contexts. Use these contexts *instead of* creating variables directly.
    # So we want to be able to indicate whether or not contexts should be allowed to
    # create variables on the fly. Within a while cond/body, they should not. Otherwise,
    # the can (e.g. when compiling a graph { ... } expression)

    proxy_names = set()
    input_keys = []
    input_values = []

    def proxy(input_map, v, placeholder_name):
      p = None
      with tf.name_scope(None):
        with tf.control_dependencies(None):
          p_name = None
          if isinstance(v, tf.Tensor) and v.dtype._is_ref_dtype:
            # eprint("creating ref proxy for", v)

            initial_value = 0
            if v.dtype.base_dtype == tf.resource:
              initial_value = None
            elif v.dtype.base_dtype == tf.string:
              initial_value = ""
            elif v.dtype.base_dtype == tf.bool:
              initial_value = False
            elif v.dtype.base_dtype == tf.float16 or v.dtype.base_dtype == tf.float32 or v.dtype.base_dtype == tf.float64:
              initial_value = 0.0

            p = tf.Variable(
                initial_value=initial_value,
                trainable=False,
                collections=[],
                name=placeholder_name,
                dtype=v.dtype.base_dtype,
                validate_shape=False)
            p.set_shape(v.get_shape())
            p_name = "%s" % p.op.name
            proxy_names.add(p_name)
            proxy_names.add("%s/read" % p.op.name)
            proxy_names.add("%s/Assign" % p.op.name)
            proxy_names.add("%s/initial_value" % p.op.name)
          elif isinstance(v, tf.Variable):
            # return (False, v)

            initial_value = 0
            if v.dtype.base_dtype == tf.resource:
              initial_value = None
            elif v.dtype.base_dtype == tf.string:
              initial_value = ""
            elif v.dtype.base_dtype == tf.bool:
              initial_value = False
            elif v.dtype.base_dtype == tf.float16 or v.dtype.base_dtype == tf.float32 or v.dtype.base_dtype == tf.float64:
              initial_value = 0.0

            p = tf.Variable(
                initial_value=initial_value,
                trainable=False,
                collections=[],
                name=placeholder_name,
                dtype=v.dtype.base_dtype,
                validate_shape=False)
            p.set_shape(v.get_shape())
            p_name = "%s:0" % p.op.name
            p = tf.get_default_graph().get_tensor_by_name(p_name)
            v = v.graph.get_tensor_by_name("%s:0" % v.op.name)
            proxy_names.add(p_name)
            proxy_names.add("%s/read" % p.op.name)
            proxy_names.add("%s/Assign" % p.op.name)
            proxy_names.add("%s/initial_value" % p.op.name)
          else:
            p = tf.placeholder(v.dtype, shape=v.get_shape(), name=placeholder_name)
            p_name = p.op.name
            proxy_names.add(p.op.name)

      # eprint("creating proxy placeholder for", self, v.graph, p.name, p, v)

      # if placeholder_name and placeholder_name != p.op.name:
      #   raise Exception("Created placeholder with unexpected name: %s vs %s" % (placeholder_name, p.op.name))

      if p_name not in input_map:
        input_map[p_name] = v
        input_keys.append(p_name)
        input_values.append(v)
      return (True, p)

    # eprint('init_exprs', init_exprs)
    initial_value_ctx = ctx.subcontext()
    initial_tensor_list = None
    initial_local_names = None
    with tf.variable_scope('while/init'):
      initial_tensor_list = [self.__unwrap_bag(self.visit(initial_value_ctx, expr)) for expr in init_exprs]
      initial_local_names = [define[1] for define in init_exprs]

    local_name_by_tensor_name = dict(zip([t.name for t in initial_tensor_list], initial_local_names))

    # We need to put placeholders in scratch graph for all init values,
    # all upvals
    proxyctx = graph_context.ProxyContext(initial_value_ctx, proxy=proxy)

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
    # input_map_shapes = [v.get_shape() for v in input_values]
    # eprint("while input_map_shapes", input_map_shapes)
    # eprint("while names to remove", proxy_names)

    # HACK(adamb) Don't actually import any nodes that are only proxies.
    #     This should probably be done automatically by the TF import
    #     logic, but empirically this is not the case.
    _while_prune(cond_meta_graph_def, proxy_names)
    _while_fix_colocations(cond_meta_graph_def, proxy_names)

    _while_prune(body_meta_graph_def, proxy_names)
    _while_fix_colocations(body_meta_graph_def, proxy_names)

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
          # eprint("while next vals", ix, t.get_shape(), t.name, local_name, body_retval_dict[local_name])
          body_retval_names.append("%s:0" % body_retval_dict[local_name])
          next_value_ixs.append(ix)
        else:
          # eprint("while next vals skipped", ix, local_name)
          pass
      else:
        # eprint("while next vals t.name", ix, t.name)
        pass

    # eprint("while initial_local_names", initial_local_names)
    # eprint("while initial_tensor_list", initial_tensor_list)
    # eprint("while input_map", input_map)
    # eprint("while body_retvals", body_retvals)
    # eprint("while body_retval_dict", body_retval_dict)
    # eprint("while body_retval_names", body_retval_names)
    # eprint("while cond_retval_name", cond_retval_name)
    # eprint("while local_name_by_tensor_name", local_name_by_tensor_name)

    def cond(*a):
      # We use a variable_scope because name_scope has a strange
      # only-sometimes-present trailing / that messes with everything.
      with tf.variable_scope('while/cond') as import_scope:
        pass
      cond_import_scope = import_scope.name

      _while_fix_context_scope(cond_meta_graph_def, cond_import_scope)

      return self._sf_while_embed(
          cond_import_scope,
          dict(zip(input_keys, a)),
          [cond_retval_name],
          cond_meta_graph_def)[0]

    def body(*a):
      body_input_map = dict(zip(input_keys, a))
      # eprint("while body", body_input_map)

      # We use a variable_scope because name_scope has a strange
      # only-sometimes-present trailing / that messes with everything.
      with tf.variable_scope('while/body') as import_scope:
        pass
      body_import_scope = import_scope.name

      _while_fix_context_scope(body_meta_graph_def, body_import_scope)

      next_values = self._sf_while_embed(
          body_import_scope,
          body_input_map,
          body_retval_names,
          body_meta_graph_def)

      body_results = list(a)
      # eprint('while body a', a)
      # eprint('while body_retval_names', body_retval_names)
      # eprint('while next_values', next_values)
      for ix, val in zip(next_value_ixs, next_values):
        val.set_shape(a[ix].get_shape())
        # eprint('while shape', ix, a[ix], a[ix].get_shape(), val, val.get_shape())
        # val.set_shape(val.get_shape())
        body_results[ix] = val

      # eprint('while body_results', body_results)
      return body_results

    # eprint("tf.while_loop(cond=%s, body=%s, loop_vars=%s)" % (cond, body, loop_vars))

    # If we're referencing variables, we need to alert listeners.
    for v in loop_vars:
      self._visit_result(v)

    results = None
    try:
      results = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars,
        parallel_iterations=1,
        back_prop=False,
      )
      # eprint("body_meta_graph_def", body_meta_graph_def)
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
  # def visit_graph_exprs(self, ctx, retval_names, exprs):
  #   for expr in exprs:
  #     result = None
  #     if expr[0] == "__retval":
  #       name = expr[1]
  #       subexpr = expr[2]
  #       result = self.visit(ctx, subexpr)
  #       ctx.define_local(name, result)
  #       retval_names.append(name)
  #     elif expr[0] == "_sf_after_leaves":
  #       # TODO(adamb) Should actually nest local variables AND leaves
  #       after_exprs = expr[1:]
  #       leaves = ctx.leaves()
  #       with tf.control_dependencies(leaves):
  #         result = self.visit_graph_exprs(ctx, retval_names, after_exprs)
  #     else:
  #       result = self.visit(ctx, expr)
  #
  #     ctx.set_above(result)

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

  def _sf_foreign_package(self, ctx, type, name, scope, path):
    eprint("_sf_foreign_package", name, scope)
    if type == "python":
      return self._sf_python_package(ctx, name, path)

    if type == "tensorflow:metagraph:pbtxt":
      eprint("_sf_tf_metagraph_package", name, scope)
      return self._sf_tf_metagraph_package(ctx, name, scope, path, binary=False)

    raise Exception("Unknown package type %s" % type)

  def _sf_tf_metagraph_package(self, ctx, name, scope, path, binary):
    eprint("_sf_tf_metagraph_package", name, scope)
    # TODO(adamb) how do we handle the fact that there may be multiple packages
    #     within the given file. Should we only parse out the one we want?
    meta_graph_def = graph_io.read_meta_graph_def(path, binary)
    pkg = graph_function.MetaGraphDefPackage(meta_graph_def, name, scope)
    ctx.define_fully_qualified_package(name, pkg)

  def _sf_python_package(self, ctx, name, path):
    outer_module = imp.new_module("%s$wrapper" % name)
    with open(path) as f:
      source = f.read()
    all_fns = self._python_importer.import_module(name, source)
    for fn_name, fn in all_fns.items():
      imported_py_func = graph_function.ImportedPythonFunction(fn)
      setattr(outer_module, fn_name, imported_py_func)

    pkg = graph_function.PythonPackage(outer_module)
    ctx.define_fully_qualified_package(name, pkg)
    return pkg

  def _sf_import(self, ctx, name_triples):
    # HACK(adamb) At the moment, assume that we're only talking about python)
    for name, package_path, scope in name_triples:
      pkg = None
      if package_path == "tensorflow":
        py_module = tf
        if scope:
          parts = scope.split("/")
          py_module = reduce(lambda p, n: getattr(p, n), parts, py_module)
        pkg = graph_function.PythonPackage(py_module)
      else:
        # TODO(adamb) Stop doing splitting in parser. Split above in python-specific code.
        pkg = ctx.fully_qualified_package(package_path)

      ctx.import_package(name, pkg)

  # HACK(adamb) For now we manually export declared functions with initial capital letters.
  #     When functions are emitted as FunctionDefs, this can be removed.
  def _maybe_export_function(self, package_name, subctx, name, value):
    if not name[0].isupper():
      eprint("not capitalized", name)
      return

    # if name.startswith("Train"):
    #   eprint("won't export training function", name)
    #   return
    #
    # if name.startswith("Test"):
    #   eprint("won't export testing function", name)
    #   return

    value = self.__unwrap_bag(value)
    eprint("considering", name)

    if not isinstance(value, graph_function.DeclaredFunction):
      eprint("isn't a declared function", type(value))
      return

    fn = value

    # HACK(adamb) We want to nest the function under "_".
    fn = fn.clone()
    fn.rename("_")

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
      fn.apply(self, subctx2, None, None, args)

      with tf.variable_scope("outputs"):
        g = tf.get_default_graph()
        for (retval_name, retval_inner_name) in fn._retval_specs():
          tensor_prefix = "%s/%s" % (package_name, name)
          try:
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

  def _sf_package(self, ctx, name, *exprs):
    pkg = ctx.resolve_fully_qualified_package(name)

    subctx = pkg.ctx()
    prev_local_items = subctx.local_items()
    prev_attr_items = subctx.attr_items()

    with tf.variable_scope(name):
      self._visit_exprs(subctx, exprs)

      for local_name, local_value in subctx.local_items():
        if (local_name, local_value) in prev_local_items:
          continue

        self._maybe_export_function(name, subctx, local_name, local_value)

      for attr_name, attr_value in subctx.attr_items():
        if (attr_name, attr_value) in prev_attr_items:
          continue

        self._maybe_export_function(name, subctx, attr_name, attr_value)

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

from tensorflow.python.ops import script_ops

def new_compilation_env():
  g = tf.Graph()
  visitor = TopLevel()
  ctx = graph_context.Context(graph_context.SentinelContextDelegate())
  ctx.import_package("tf", graph_function.PythonPackage(tf))
  ctx.import_package("nao", graph_function.PythonPackage(Nao(visitor), prepend_with_context=True))

  return g, visitor, ctx

def meta_graph_def_from_exprs(exprs):
  g, visitor, ctx = new_compilation_env()

  with g.as_default():
    visitor._visit_exprs(ctx, exprs)

    # NOTE(adamb) Could also store files to copy out in assets_collection
    py_func_data = visitor._python_importer.dump_py_funcs(script_ops._py_funcs)
    js_py_func_data = tf.constant(json.dumps(py_func_data), name="py_funcs_json")

    meta_graph_def, _ = meta_graph.export_scoped_meta_graph()
    return meta_graph_def
