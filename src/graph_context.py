import copy
import sys

import tensorflow as tf

from collections import OrderedDict

import graph_function

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class SentinelContextDelegate:
  def __init__(self):
    self._delegate = None

  def call(self, fn, args, kwargs):
    try:
      return fn(*args, **kwargs)
    except:
      raise Exception("Tried to call %s with args %s and kwargs %s"  % (fn, args, kwargs))

  def get_index(self, target, index):
    # eprint("get_index", target, index, type(index))

    if type(target) == graph_function.RetvalBag:
      return target.get(index)

    if isinstance(index, int):
      if hasattr(target, '__getitem__'):
        return target[index]

    # HACK(adamb) This will return an wrapped python function.
    # We're using it at the moment to get access to op-producing
    # instance methods like those on QueueBase. 8[
    v = getattr(target, index)
    if callable(v):
      return graph_function.PrimitiveFunction(v)

    return v

  def fully_qualified_package(self, name):
    raise Exception("Package not available: %s" % name)

  def imported_package(self, name):
    raise Exception("Package not imported: %s" % name)

  def get_attr(self, name):
    raise Exception("No such attribute: %s" % name)

  def get_local(self, name):
    raise Exception("No such local or function: %s" % name)

  def eliminate_leaf(self, op):
    pass

  def duplicate_for(self, other):
    return other.duplicate()

  def leaves(self):
    return frozenset([])

class ProxyContext:
  def __init__(self, other,
      proxy=None,
      input_map=None,
      input_keys=None,
      input_values=None,
      proxy_names=None,
      name_cache=None):
    self._proxy = proxy
    self._placeholder_op_cache = {}
    self._other = other

    if input_map == None:
      input_map = OrderedDict()
    self._input_map = input_map

    if name_cache == None:
      name_cache = {}
    self._placeholder_name_cache = name_cache

  def duplicate_for(self, other):
    d = other.duplicate()
    delegate = d._delegate
    while delegate:
      if type(delegate) == ProxyContext:
        return d

      delegate = delegate._delegate

    d._delegate = ProxyContext(d._delegate,
        proxy=self._proxy,
        input_map=self._input_map)
    return d

  def clear_placeholder_op_cache(self):
    self._placeholder_op_cache = {}

  def subcontext(self):
    return Context(self)

  def input_map(self):
    return OrderedDict(self._input_map)

  def call(self, fn, args, kwargs):
    # Operations on queues mean we have to do gross stuff like inspect return
    # values of python functions.
    try:
      # return fn(*args, **kwargs)
      r = fn(*args, **kwargs)
      # r = self._proxy(r, None)[1]
      return r
    except:
      raise Exception("Tried to call %s with args %s and kwargs %s"  % (fn, args, kwargs))

  def _maybe_proxy(self, v):
    t = type(v)
    if t != tf.Operation and t != tf.Tensor and t != tf.Variable:
      return v

    if v.graph == tf.get_default_graph():
      return v

    return self._proxy(self._input_map, v, None)[1]

  def get_index(self, target, index):
    return self._maybe_proxy(self._other.get_index(target, index))

  def fully_qualified_package(self, name):
    return self._other.fully_qualified_package(name)

  def imported_package(self, name):
    return self._other.imported_package(name)

  def get_attr(self, name):
    return self._other.get_attr(name)

  def get_local(self, name):
    if type(name) != str:
      raise Exception("Can't get_local for non string: %s" % name)

    if name in self._placeholder_op_cache:
      return self._placeholder_op_cache[name]

    placeholder_name = None
    if name in self._placeholder_name_cache:
      placeholder_name = self._placeholder_name_cache[name]
    else:
      placeholder_name = "Proxy_%d" % len(self._input_map)

    did_proxy, v = self._proxy(self._input_map, self._other.get_local(name), placeholder_name)
    if did_proxy:
      if name not in self._placeholder_name_cache:
        self._placeholder_name_cache[name] = v.op.name
      self._placeholder_op_cache[name] = v
    return v

  def define_local(self, n, v):
    return self._other.define_local(n, v)

  def set_above(self, v):
    return self._other.set_above(v)

  def possible_leaf(self, op):
    return self._other.possible_leaf(op)

  def eliminate_leaf(self, op):
    return self._other.eliminate_leaf(op)

  def leaves(self):
    return self._other.leaves()

class Context:
  def __init__(self, delegate):
    self._delegate = delegate
    self._fully_qualified_packages = {}
    self._imported_packages = {}
    self._attrs = {}
    self._locals = {}
    self._leaves = set()
    self._above = None

  def duplicate_for(self, other):
    return self._delegate.duplicate_for(other)

  def call(self, fn, args, kwargs):
    return self._delegate.call(fn, args, kwargs)

  def get_index(self, target, index):
    return self._delegate.get_index(target, index)

  def subcontext(self):
    return Context(self)

  def define_fully_qualified_package(self, name, pkg):
    if name in self._fully_qualified_packages:
      raise Exception("Already defined package: %s" % name)

    self._fully_qualified_packages[name] = pkg

  def fully_qualified_package(self, name):
    if name in self._fully_qualified_packages:
      return self._fully_qualified_packages[name]

    return self._delegate.fully_qualified_package(name)

  def import_package(self, name, pkg):
    if name in self._imported_packages:
      raise Exception("Already imported package: %s" % name)

    self._imported_packages[name] = pkg

  def imported_package(self, name):
    if name in self._imported_packages:
      return self._imported_packages[name]

    return self._delegate.imported_package(name)

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

    # If a local is *completely constant*, we can treat it like an attr.
    if name in self._locals:
      local = self._locals[name]
      if isinstance(local, tf.Tensor):
        try:
          return tf.contrib.util.constant_value(local)
        except TypeError as te:
          raise Exception("Local is not usable as an attribute: " + name)
      return local

    return self._delegate.get_attr(name)

  def get_local(self, name):
    if type(name) != str:
      raise Exception("Tried to look up local with non-string name: %s" % name)

    if name == '^':
      return self._above

    if name in self._locals:
      return self._locals[name]

    if name in self._attrs:
      return self._attrs[name]

    return self._delegate.get_local(name)

  def get_local_strict(self, name):
    if name in self._locals:
      return self._locals[name]

    raise Exception("No such entry: %s. Have: %s" % (name, self._locals))

  def possible_leaf(self, op):
    t = type(op)
    if t == tf.Tensor or t == tf.Operation:
      self._leaves.add(op)

    if t == graph_function.RetvalBag:
      for v in op.values():
        self.possible_leaf(v)

  def eliminate_leaf(self, op):
    t = type(op)
    if t == tf.Tensor or t == tf.Operation:
      self._leaves.discard(op)

    return self._delegate.eliminate_leaf(op)

  def leaves(self):
    l = frozenset(self._leaves)
    # if self._delegate:
    #   l = l | self._delegate.leaves()
    return l

  def __str__(self):
    return "%s" % self._locals
