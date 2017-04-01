from nao.compiler.primitive_function import PrimitiveFunction

class PythonPackage:
  def __init__(self, mod, prepend_with_context=False):
    self._mod = mod
    self._prepend_with_context = prepend_with_context

  def apply(self, visitor, ctx, name, attrs, args):
    # eprint("applying %s to args %s" % (self, args))
    n, *_ = args
    if n.startswith('__'):
      raise Exception("Tried to use non-exported namespace entry named: %s" % n)

    val = getattr(self._mod, n)
    if callable(val):
      return PrimitiveFunction(val, self._prepend_with_context)

    return val
