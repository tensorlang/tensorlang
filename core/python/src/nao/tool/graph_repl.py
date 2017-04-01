import tensorflow as tf

from nao.compiler.retvalbag import RetvalBag

class ReplSession:
  def __init__(self, compiler):
    self._suffix = ".nao"
    self._compiler = compiler
    self._session = compiler.new_session()

  def run(self, src):
    self._compiler.clear()
    self._compiler.put_source("main%s" % self._suffix, src)

    pkg = self._compiler.resolve_import_path("main", reimport=True)
    above = pkg.ctx().get_above()

    if isinstance(above, RetvalBag):
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
    except KeyboardInterrupt:
      print("^C")
      continue
    except EOFError:
      print()
      break

    try:
      result = repl_session.run(src)
    except Exception as e:
      result = e

    print(result)
