import copy
import sys

import tensorflow as tf
from tensorflow.python.ops import state_ops

from collections import OrderedDict

from nao.compiler.primitive_function import PrimitiveFunction
from nao.compiler.retvalbag import RetvalBag
from nao.compiler.nao import graph_function

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class SentinelContextDelegate:
  def __init__(self):
    self._delegate = None

  def local_items(self):
    return []

  def attr_items(self):
    return []

  def call(self, fn, args, kwargs):
    try:
      return fn(*args, **kwargs)
    except:
      raise Exception("Tried to call %s with args %s and kwargs %s"  % (fn, args, kwargs))

  def get_index(self, target, index):
    # eprint("get_index", target, index, type(index))

    if type(target) == RetvalBag:
      return target.get(index)

    if isinstance(index, int):
      if hasattr(target, '__getitem__'):
        return target[index]

    # HACK(adamb) This will return an wrapped python function.
    # We're using it at the moment to get access to op-producing
    # instance methods like those on QueueBase. 8[
    v = getattr(target, index)
    if callable(v):
      return PrimitiveFunction(v)

    return v

  def fully_qualified_package(self, name):
    raise Exception("Package not available: %s" % name)

  def imported_package(self, name):
    raise Exception("Package not imported: %s" % name)

  def get_attr(self, name):
    raise Exception("No such attribute: %s" % name)

  def update_local(self, name, rhs):
    raise Exception("No such local or function: %s" % name)

  def get_local(self, name):
    raise Exception("No such local or function: %s" % name)

  def eliminate_leaf(self, op):
    pass

  def duplicate_for(self, other):
    return other.duplicate()

  def leaves(self):
    return frozenset([])

class Context:
  def __init__(self, delegate, proxy=None):
    self._proxy = proxy
    self._delegate = delegate
    self._fully_qualified_packages = {}
    self._imported_packages = {}
    self._wrap_locals_in_vars = False
    self._attrs = {}
    self._locals = {}
    self._leaves = set()
    self._above = None

  def wrap_locals_in_vars(self):
    self._wrap_locals_in_vars = True

  def duplicate_for(self, other):
    ctx = other.duplicate()
    ctx._proxy = self._proxy
    return ctx

  def call(self, fn, args, kwargs):
    return self._maybe_proxy(self._delegate.call(fn, args, kwargs))

  def get_index(self, target, index):
    return self._maybe_proxy(self._delegate.get_index(target, index))

  def subcontext(self):
    return Context(self, proxy=self._proxy)

  def local_items(self):
    l = self._delegate.local_items()
    l.extend(self._locals.items())
    return l

  def attr_items(self):
    l = self._delegate.attr_items()
    l.extend(self._attrs.items())
    return l

  def resolve_fully_qualified_package(self, name):
    if name in self._fully_qualified_packages:
      return self._fully_qualified_packages[name]

    subctx = self.subcontext()
    pkg = graph_function.Package(subctx)
    self.define_fully_qualified_package(name, pkg)
    return pkg

  def define_fully_qualified_package(self, name, pkg):
    if name in self._fully_qualified_packages:
      raise Exception("Already defined package: %s" % name)

    eprint("Defining package", name)
    self._fully_qualified_packages[name] = pkg

  def fully_qualified_package(self, name):
    if name in self._fully_qualified_packages:
      return self._fully_qualified_packages[name]

    return self._delegate.fully_qualified_package(name)

  def import_package(self, name, pkg):
    if name in self._imported_packages:
      raise Exception("Already imported package: %s" % name)

    eprint("Importing package", name)
    self._imported_packages[name] = pkg

  def imported_package(self, name):
    if name in self._imported_packages:
      return self._imported_packages[name]

    return self._delegate.imported_package(name)

  def duplicate(self):
    ctx = copy.copy(self)
    ctx._attrs = copy.copy(ctx._attrs)
    ctx._locals = copy.copy(ctx._locals)
    ctx._leaves = copy.copy(ctx._leaves)
    return ctx

  def get_above(self):
    return self._maybe_proxy(self._above)

  def set_above(self, value):
    self._above = value

  def _maybe_proxy(self, v):
    if self._proxy is None:
      return v

    return self._proxy(v)

  def update_local(self, name, rhs):
    if name in self._locals:
      v = self._locals[name]
      if not isinstance(v, tf.Variable) and not (isinstance(v, tf.Tensor) and v.dtype._is_ref_dtype):
        raise Exception("%s not a variable: %s" % (name, v))
      eprint("updating local", name, "from", v, "to", rhs)
      v = tf.assign(v, rhs)
      eprint("updated local", name, "is", v)
      self._locals[name] = v
      return v

    return self._delegate.update_local(name, rhs)

  def define_local(self, name, value):
    if name in self._locals:
      raise Exception("Local already defined: %s" % name)

    should_wrap_in_var = False
    if self._wrap_locals_in_vars:
      if isinstance(value, tf.Tensor):
        should_wrap_in_var = True

      # HACK(adamb) Unwrapping in here really isn't great, since auto-unwrapping can create unexpected behavior.
      if isinstance(value, RetvalBag) and value.len() == 1:
        if isinstance(value.get(None), tf.Tensor):
          should_wrap_in_var = True
          value = value.get(None)

    if should_wrap_in_var:
      variable = state_ops.variable_op_v2(
          value.get_shape(),
          value.dtype.base_dtype)

      with tf.control_dependencies(None):
        value = tf.identity(
          tf.cond(
            tf.is_variable_initialized(variable),
            lambda: variable,
            lambda: tf.assign(variable, value)
          )
        )
      print("value", value)

    self._locals[name] = value
    return value

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
      return self._maybe_proxy(self._attrs[name])

    # If a local is *completely constant*, we can treat it like an attr.
    if name in self._locals:
      local = self._locals[name]
      if isinstance(local, tf.Tensor):
        try:
          return self._maybe_proxy(tf.contrib.util.constant_value(local))
        except TypeError as te:
          raise Exception("Local is not usable as an attribute: " + name)
      return self._maybe_proxy(local)

    return self._maybe_proxy(self._delegate.get_attr(name))

  def get_local(self, name):
    if type(name) != str:
      raise Exception("Tried to look up local with non-string name: %s" % name)

    if name == '^':
      return self._maybe_proxy(self._above)

    if name in self._locals:
      return self._maybe_proxy(self._locals[name])

    if name in self._attrs:
      return self._maybe_proxy(self._attrs[name])

    if name in self._imported_packages:
      return self._maybe_proxy(self._imported_packages[name])

    return self._maybe_proxy(self._delegate.get_local(name))

  def get_local_strict(self, name):
    if name in self._locals:
      return self._maybe_proxy(self._locals[name])

    raise Exception("No such entry: %s. Have: %s" % (name, self._locals))

  def possible_leaf(self, v):
    if isinstance(v, (tf.Tensor, tf.Operation, tf.Variable)):
      self._leaves.add(v)

    if isinstance(v, RetvalBag):
      for v_ in v.values():
        self.possible_leaf(v_)

  def eliminate_leaf(self, v):
    if isinstance(v, (tf.Tensor, tf.Operation, tf.Variable)):
      self._leaves.discard(v)

    if isinstance(v, RetvalBag):
      for v_ in v.values():
        self.eliminate_leaf(v_)
        return

    return self._delegate.eliminate_leaf(v)

  def leaves(self):
    l = frozenset(self._leaves)
    # if self._delegate:
    #   l = l | self._delegate.leaves()
    return l

  def __str__(self):
    return "%s -> %s" % (self._locals.items(), self._delegate)
