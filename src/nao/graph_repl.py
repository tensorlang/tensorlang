import tensorflow as tf

from nao import graph_gen
from nao import graph_function

class ReplSession:
  def __init__(self, parser):
    self._parser = parser
    g, visitor, ctx = graph_gen.new_compilation_env()
    self._g = g
    self._visitor = visitor
    self._ctx = ctx
    self._session = tf.Session()

  def run(self, src):
    self._parser.clear()
    self._parser.put_source("main", src)
    self._parser.resolve_import("main", None)
    exprs = self._parser.pallet()
    last_expr_result = self._visitor._visit_exprs(self._ctx, exprs)

    pkg = self._ctx.fully_qualified_package("main")
    c = pkg.ctx()

    above = c.get_above()
    if isinstance(above, graph_function.RetvalBag):
      above = above.get(None)

    if isinstance(above, (tf.Tensor, tf.Variable, tf.Operation)):
      return above.eval(session=self._session)

  def __del__(self):
    self._session.close()

import atexit
import os
import readline

HISTORY_BASENAME = '.nao_history'

def run(parser):
  histfile = os.path.join(os.path.expanduser("~"), HISTORY_BASENAME)

  try:
    readline.read_history_file(histfile)
    h_len = readline.get_history_length()
  except FileNotFoundError:
    open(histfile, 'wb').close()
    h_len = 0

  def save(prev_h_len, histfile):
    new_h_len = readline.get_history_length()
    readline.set_history_length(1000)
    readline.append_history_file(new_h_len - prev_h_len, histfile)

  repl_session = ReplSession(parser)
  while True:
    try:
      src = input("> ")
      if src == "":
        continue
    except EOFError:
      break

    print(repl_session.run(src))
