import inspect

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

  def _name(self):
    return self._fn.__name__

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
