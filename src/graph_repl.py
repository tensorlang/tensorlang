import atexit
import os
import readline
import subprocess
import json

import tensorflow as tf

import graph_gen
import graph_function

HISTORY_BASENAME = '.nao_history'


def run():
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

  with tf.Session() as sess:
    g, visitor, ctx = graph_gen.new_compilation_env()

    while True:
      try:
        src = input("> ")
        if src == "":
          continue
      except EOFError:
        break

      with subprocess.Popen(
          ["node", "lib/cli.js", "--source", src, "--parse=-"],
          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        expr_text = proc.stdout.read().decode('utf-8')
      exprs = json.loads(expr_text)
      last_expr_result = visitor._visit_exprs(ctx, exprs)

      pkg = ctx.fully_qualified_package("main")
      c = pkg.ctx()

      # Actually need to grab all the *leaves*
      # print(exprs)
      # print(c.leaves())
      # print(last_expr_result)
      above = c.get_above()
      if isinstance(above, graph_function.RetvalBag):
        above = above.get(None)

      if isinstance(above, (tf.Tensor, tf.Variable, tf.Operation)):
        r = above.eval()
        print(r)

    # meta_graph_def = graph_gen.meta_graph_def_from_exprs(exprs)
    # print(meta_graph_def)

    # TODO(adamb) Support import
    # TODO(adamb) Support literal tensors
    # TODO(adamb) Support defining functions
    # TODO(adamb) Support macros

    # Parse string -> sexpr
    # Compile sexpr
    # Run sexpr
    # print(src)
