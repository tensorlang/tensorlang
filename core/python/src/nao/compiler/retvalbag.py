import tensorflow as tf

def _graph_for(v):
  if isinstance(v, (tf.Operation, tf.Tensor, tf.Variable)):
    return v.graph
  return None

def unwrap_bag(v):
  if type(v) == RetvalBag:
    v = v.get(None)
  return v


class RetvalBag:
  def __init__(self, a_dict, fn=None):
    self._d = {}
    self.graph = None

    for k, v in a_dict.items():
      if type(v) == RetvalBag:
        v = v.get(None)
      if fn:
        v = fn(v)
      if self.graph is None:
        self.graph = _graph_for(v)
      self._d[k] = v

    for k, v in self._d.items():
      g = _graph_for(v)
      if g is not None and self.graph != g:
        graphs = [(k, v, _graph_for(v)) for k, v in self._d.items()]
        raise Exception("RetvalBag can only contain elements from a single graph. Got elements from multiple graphs: %s" % graphs)

  def get(self, key):
    if key == None:
      key = self._default_key()
    return self._d[key]

  def values(self):
    return self._d.values()

  def wrap(self, fn):
    return RetvalBag(self._d, fn=fn)

  def len(self):
    return len(self._d)

  def __str__(self):
    return "RetvalBag(%s)" % self._d

  def _default_key(self):
    l = len(self._d)
    if l == 0:
      raise Exception("Can't get default retval for an empty RetvalBag")
    if l > 1:
      raise Exception("Can't get default retval a RetvalBag with more than one entry: %s" % self._d)
    return list(self._d.keys())[0]
