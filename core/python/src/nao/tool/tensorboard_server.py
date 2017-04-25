import logging as base_logging
import os
import socket
import sys
from werkzeug import serving

from tensorflow.python.platform import app
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.tensorboard.backend import application

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)


def make_simple_server(tb_app, host, port):
  """Create an HTTP server for TensorBoard.

  Args:
    tb_app: The TensorBoard WSGI application to create a server for.
    host: Indicates the interfaces to bind to ('::' or '0.0.0.0' for all
        interfaces, '::1' or '127.0.0.1' for localhost). A blank value ('')
        indicates protocol-agnostic all interfaces.
    port: The port to bind to (0 indicates an unused port selected by the
        operating system).
  Returns:
    A tuple of (server, url):
      server: An HTTP server object configured to host TensorBoard.
      url: A best guess at a URL where TensorBoard will be accessible once the
        server has been started.
  Raises:
    socket.error: If a server could not be constructed with the host and port
      specified. Also logs an error message.
  """
  # Mute the werkzeug logging.
  base_logging.getLogger('werkzeug').setLevel(base_logging.WARNING)

  try:
    if host:
      # The user gave us an explicit host
      server = serving.make_server(host, port, tb_app, threaded=True)
      if ':' in host and not host.startswith('['):
        # Display IPv6 addresses as [::1]:80 rather than ::1:80
        final_host = '[{}]'.format(host)
      else:
        final_host = host
    else:
      # We've promised to bind to all interfaces on this host. However, we're
      # not sure whether that means IPv4 or IPv6 interfaces.
      try:
        # First try passing in a blank host (meaning all interfaces). This,
        # unfortunately, defaults to IPv4 even if no IPv4 interface is available
        # (yielding a socket.error).
        server = serving.make_server(host, port, tb_app, threaded=True)
      except socket.error:
        # If a blank host didn't work, we explicitly request IPv6 interfaces.
        server = serving.make_server('::', port, tb_app, threaded=True)
      final_host = socket.gethostname()
    server.daemon_threads = True
  except socket.error as socket_error:
    if port == 0:
      msg = 'TensorBoard unable to find any open port'
    else:
      msg = (
          'TensorBoard attempted to bind to port %d, but it was already in use'
          % port)
    logging.error(msg)
    print(msg)
    raise socket_error

  final_port = server.socket.getsockname()[1]
  tensorboard_url = 'http://%s:%d' % (final_host, final_port)
  return server, tensorboard_url


def run_simple_server(tb_app, host, port):
  """Run a TensorBoard HTTP server, and print some messages to the console."""
  try:
    server, url = make_simple_server(tb_app, host, port)
  except socket.error:
    # An error message was already logged
    exit(-1)
  msg = 'Starting TensorBoard %s at %s' % (tb_app.tag, url)
  print(msg)
  logging.info(msg)
  print('(Press CTRL+C to quit)')
  sys.stdout.flush()

  server.serve_forever()

def main(tb_logdir, tb_host=None, tb_port=None, tb_debug=True, tb_purge_orphaned_data=False, tb_reload_interval=5):
  if tb_host is None:
    tb_host = '127.0.0.1'

  if tb_port is None:
    tb_port = 6006

  tb_port = int(tb_port or '6006')

  tb = application.standard_tensorboard_wsgi(
      logdir=tb_logdir,
      purge_orphaned_data=tb_purge_orphaned_data,
      reload_interval=tb_reload_interval)

  run_simple_server(tb, host=tb_host, port=tb_port)
