import sys

import tensorflow as tf
from tensorflow.core.protobuf import control_flow_pb2

from nao.compiler.retvalbag import RetvalBag, unwrap_bag

from collections import OrderedDict

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def zero_value_for_dtype(dtype):
  value = 0
  if dtype.base_dtype == tf.resource:
    value = None
  elif dtype.base_dtype == tf.string:
    value = ""
  elif dtype.base_dtype == tf.bool:
    value = False
  elif dtype.base_dtype == tf.float16 or dtype.base_dtype == tf.float32 or dtype.base_dtype == tf.float64:
    value = 0.0
  return value

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

def _while_fix_colocations(meta_graph_def, proxy_cruft):
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
          if op_name not in proxy_cruft and "%s:0" % op_name not in proxy_cruft:
            # eprint("Skipping replacement of", op_name, "not in", list(proxy_cruft))
            continue
          # replacement_name = proxy_cruft[op_name].name
          # replacement = compat.as_bytes("loc:@" + replacement_name)
          # eprint("Replacing", class_value, "with", replacement)
          # class_values[ix] = replacement
          # HACK(adamb) It would be much, much better to just do the replacement
          #     commented out above, but we apparently can't replace a location
          #     with a value pointing to the existing graph. Strange.
          eprint("HACK(adamb) Removing", class_value, "for op_name", op_name, "node name", node.name)
          del class_values[ix]

def _sf_while_inner(use_device, visitor_class, ctx, exprs):
  with tf.Graph().as_default() as g:
    with tf.device(use_device):
      visitor = visitor_class()
      final_tensor = visitor._visit_exprs(ctx, exprs)

      # We do not want to include shapes, since inferred shapes will cause problems
      # for shape inference upon import and re-export.

      # HACK(adamb) Since TensorFlow uses __del__ to clean up py_funcs, we need to copy them.
      cleanup_py_funcs_used_in_graph = []
      if hasattr(g, "_cleanup_py_funcs_used_in_graph"):
        cleanup_py_funcs_used_in_graph = g._cleanup_py_funcs_used_in_graph[:]

      return (
        tf.train.export_meta_graph(),
        cleanup_py_funcs_used_in_graph,
        unwrap_bag(final_tensor).name
      )

def _sf_while_embed(import_scope, input_map, retval_names, meta_graph_def, cleanup_py_funcs):
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

    if not hasattr(g, "_cleanup_py_funcs_used_in_graph"):
      g._cleanup_py_funcs_used_in_graph = []

    g._cleanup_py_funcs_used_in_graph.extend(cleanup_py_funcs)

    # eprint("have graph", import_scope, g.as_graph_def(add_shapes=True))

    return [g.get_tensor_by_name("%s/%s" % (import_scope, n)) for n in retval_names]
  except KeyError as ke:
    eprint('error, but got graph', tf.get_default_graph().as_graph_def())
    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    nodes.sort()
    eprint('error, but got nodes', nodes)
    raise ke

def _sf_while_loop(visitor, ctx, cond_expr, body_exprs, body_retvals, init_exprs):
  # Need to evaluate body_exprs first, looking for all variables that will be created
  # internally. Roll up into nested variable contexts. Unroll these contexts to be
  # passed as part of var_list. Within def body(*a), repackage these variables into
  # variable contexts. Use these contexts *instead of* creating variables directly.
  # So we want to be able to indicate whether or not contexts should be allowed to
  # create variables on the fly. Within a while cond/body, they should not. Otherwise,
  # the can (e.g. when compiling a graph { ... } expression)

  # track these so we can eventually remove them.
  proxy_cruft = set()
  proxied_placeholder_names = OrderedDict()
  proxied_placeholders = OrderedDict()

  # track replacements. focus on external entity to internal name.
  def proxy(v):
    # eprint("proxy", v)

    if isinstance(v, RetvalBag):
      if v.graph is None:
        return v

      if v.graph == tf.get_default_graph():
        return v

      return v.wrap(proxy)

    if not isinstance(v, (tf.Operation, tf.Tensor, tf.Variable)):
      return v

    if v.graph == tf.get_default_graph():
      return v

    if v.name in proxied_placeholders:
      return proxied_placeholders[v.name]

    if v.name in proxied_placeholder_names:
      placeholder_name = proxied_placeholder_names[v.name]
    else:
      placeholder_name = "Proxy_%d" % len(proxied_placeholder_names)

    p = None
    with tf.name_scope(None):
      with tf.control_dependencies(None):
        p_name = None
        if isinstance(v, tf.Tensor) and v.dtype._is_ref_dtype:
          p = tf.Variable(
              initial_value=zero_value_for_dtype(v.dtype),
              trainable=False,
              collections=[],
              name=placeholder_name,
              dtype=v.dtype.base_dtype,
              validate_shape=False)
          p.set_shape(v.get_shape())
          p_name = "%s" % p.op.name
          proxy_cruft.add(p_name)
          proxy_cruft.add("%s/read" % p.op.name)
          proxy_cruft.add("%s/Assign" % p.op.name)
          proxy_cruft.add("%s/initial_value" % p.op.name)
        elif isinstance(v, tf.Variable):
          p = tf.Variable(
              initial_value=zero_value_for_dtype(v.dtype),
              trainable=False,
              collections=[],
              name=placeholder_name,
              dtype=v.dtype.base_dtype,
              validate_shape=False)
          p.set_shape(v.get_shape())
          p_name = "%s:0" % p.op.name
          p = tf.get_default_graph().get_tensor_by_name(p_name)
          v = v.graph.get_tensor_by_name("%s:0" % v.op.name)
          proxy_cruft.add(p_name)
          proxy_cruft.add("%s/read" % p.op.name)
          proxy_cruft.add("%s/Assign" % p.op.name)
          proxy_cruft.add("%s/initial_value" % p.op.name)
        else:
          p = tf.placeholder(
              v.dtype,
              shape=v.get_shape(),
              name=placeholder_name)
          p_name = p.op.name
          proxy_cruft.add(p_name)

    proxied_placeholders[v.name] = p
    proxied_placeholder_names[v.name] = p_name

    if placeholder_name and placeholder_name != p.op.name:
      raise Exception("Created placeholder with unexpected name: %s vs %s" % (placeholder_name, p.op.name))

    return p

  g = tf.get_default_graph()
  while_loop_name = g.unique_name("while", False)

  # eprint('init_exprs', init_exprs)
  initial_value_ctx = ctx.subcontext()
  initial_value_ctx._proxy = proxy
  initial_tensor_list = None
  initial_local_names = None
  with tf.variable_scope('%s_init' % while_loop_name):
    initial_tensor_list = [unwrap_bag(visitor.visit(initial_value_ctx, expr)) for expr in init_exprs]
    initial_local_names = [define[1] for define in init_exprs]

  local_name_by_tensor_name = dict(zip([t.name for t in initial_tensor_list], initial_local_names))

  device_stack = g._device_function_stack
  use_device = None
  if len(device_stack) > 0:
    use_device = device_stack[-1]

  eprint("Will use device", use_device)

  # Ensure we have a placeholder for every initial value.
  with tf.Graph().as_default():
    with tf.device(use_device):
      for local_name in initial_local_names:
        initial_value_ctx.get_local(local_name)

  # Don't let cached placeholders from init_exprs infect our graph.
  proxied_placeholders = OrderedDict()
  cond_ctx = initial_value_ctx.subcontext()
  cond_meta_graph_def, cond_cleanup_funcs, cond_retval_name = _sf_while_inner(use_device, type(visitor), cond_ctx, [cond_expr])

  # Don't let cached placeholders from cond_exprs infect our graph.
  proxied_placeholders = OrderedDict()
  body_ctx = initial_value_ctx.subcontext()
  body_meta_graph_def, body_cleanup_funcs, _ = _sf_while_inner(use_device, type(visitor), body_ctx, body_exprs)

  # HACK(adamb) Don't actually import any nodes that are only proxies.
  #     This should probably be done automatically by the TF import
  #     logic, but empirically this is not the case.
  _while_prune(cond_meta_graph_def, proxy_cruft)
  _while_fix_colocations(cond_meta_graph_def, proxy_cruft)

  _while_prune(body_meta_graph_def, proxy_cruft)
  _while_fix_colocations(body_meta_graph_def, proxy_cruft)

  body_retval_dict = dict(body_retvals)
  body_retval_names = []
  next_value_ixs = []

  loop_vars = [g.get_tensor_by_name(v_name) for v_name in proxied_placeholder_names.keys()]

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
  # eprint("while proxied_placeholder_names", proxied_placeholder_names)
  # eprint("while local_name_by_tensor_name", local_name_by_tensor_name)

  def cond(*a):
    # We use a variable_scope because name_scope has a strange
    # only-sometimes-present trailing / that messes with everything.
    cond_import_scope = '%s_cond' % while_loop_name

    _while_fix_context_scope(cond_meta_graph_def, cond_import_scope)

    return _sf_while_embed(
        cond_import_scope,
        dict(zip(proxied_placeholder_names.values(), a)),
        [cond_retval_name],
        cond_meta_graph_def,
        cond_cleanup_funcs)[0]

  def body(*a):
    body_input_map = dict(zip(proxied_placeholder_names.values(), a))
    # eprint("while body", body_input_map)

    # We use a variable_scope because name_scope has a strange
    # only-sometimes-present trailing / that messes with everything.
    body_import_scope = '%s_body' % while_loop_name

    _while_fix_context_scope(body_meta_graph_def, body_import_scope)

    next_values = _sf_while_embed(
        body_import_scope,
        body_input_map,
        body_retval_names,
        body_meta_graph_def,
        body_cleanup_funcs)

    body_results = list(a)
    for ix, val in zip(next_value_ixs, next_values):
      val.set_shape(a[ix].get_shape())
      # eprint('while shape', ix, a[ix], a[ix].get_shape(), val, val.get_shape())
      # val.set_shape(val.get_shape())
      body_results[ix] = val

    # eprint('while body_results', body_results)
    return body_results

  # If we're referencing variables, we need to alert listeners.
  for v in loop_vars:
    visitor._visit_result(v)

  results = None
  results = tf.while_loop(
    cond=cond,
    body=body,
    loop_vars=loop_vars,
    parallel_iterations=1,
    back_prop=False,
    name=while_loop_name.split("/")[-1],
  )

  if type(results) != list:
    results = [results]

  r = {}
  for k_name, v in zip(proxied_placeholder_names.keys(), results):
    if k_name in local_name_by_tensor_name:
      r[local_name_by_tensor_name[k_name]] = v

  return RetvalBag(r)
