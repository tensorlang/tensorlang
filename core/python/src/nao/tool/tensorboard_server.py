import socket
import sys

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import status_bar
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_file_inspector as efi
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import server

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def main(tb_logdir, tb_host=None, tb_port=None, tb_debug=True, tb_purge_orphaned_data=False, tb_reload_interval=5):
  if tb_host is None:
    tb_host = '127.0.0.1'

  if tb_port is None:
    tb_port = 6006

  tb_port = int(tb_port or '6006')

  if tb_debug:
    logging.set_verbosity(logging.DEBUG)
    logging.info('TensorBoard is in debug mode.')

  path_to_run = server.ParseEventFilesSpec(tb_logdir)
  logging.info('TensorBoard path_to_run is: %s', path_to_run)

  multiplexer = event_multiplexer.EventMultiplexer(
      size_guidance=server.TENSORBOARD_SIZE_GUIDANCE,
      purge_orphaned_data=tb_purge_orphaned_data)
  server.StartMultiplexerReloadingThread(multiplexer, path_to_run,
                                         tb_reload_interval)
  try:
    tb_server = server.BuildServer(multiplexer, tb_host, tb_port, tb_logdir)
  except socket.error:
    if tb_port == 0:
      msg = 'Unable to find any open ports.'
      logging.error(msg)
      eprint(msg)
      return -2
    else:
      msg = 'Tried to connect to port %d, but address is in use.' % tb_port
      logging.error(msg)
      eprint(msg)
      return -3

  try:
    tag = resource_loader.load_resource('tensorboard/TAG').strip()
    logging.info('TensorBoard is tag: %s', tag)
  except IOError:
    logging.info('Unable to read TensorBoard tag')
    tag = ''

  status_bar.SetupStatusBarInsideGoogle('TensorBoard %s' % tag, tb_port)
  eprint('Starting TensorBoard %s on port %d' % (tag, tb_port))
  eprint('(You can navigate to http://%s:%d)' % (tb_host, tb_port))

  tb_server.serve_forever()
