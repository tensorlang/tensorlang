# Originally based on simple_kernel.py
# by Doug Blank <doug.blank@gmail.com>
#
# To adjust debug output, set debug_level to:
#  0 - show no debugging information
#  1 - shows basic running information
#  2 - also shows loop details
#  3 - also shows message details
#
from __future__ import print_function

## General Python imports:
import sys
import os

import json
import hmac
import uuid
import errno
import hashlib
import datetime
import threading
from pprint import pformat

# zmq specific imports:
import zmq
from zmq.eventloop import ioloop, zmqstream
from zmq.error import ZMQError

PYTHON3 = sys.version_info.major == 3

#Globals:
DELIM = b"<IDS|MSG>"

debug_level = 1 # 0 (none) to 3 (all) for various levels of detail
def dprint(level, *args, **kwargs):
  """ Show debug information """
  if level <= debug_level:
    print("DEBUG:", *args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

class WireProtocol:
  def __init__(self, engine_id, secure_key, signature_scheme):
    self._engine_id = engine_id
    signature_schemes = {"hmac-sha256": hashlib.sha256}
    self._auth = hmac.HMAC(
      self._str_to_bytes(secure_key),
      digestmod=signature_schemes[signature_scheme])

  def _str_to_bytes(self, s):
    return s.encode('ascii') if PYTHON3 else bytes(s)

  def _msg_id(self):
    """ Return a new uuid for message id """
    return str(uuid.uuid4())

  def _new_header(self, msg_type):
    """make a new header"""
    return {
        "date": datetime.datetime.now().isoformat(),
        "msg_id": self._msg_id(),
        "username": "kernel",
        "session": self._engine_id,
        "msg_type": msg_type,
        "version": "5.0",
      }

  def sign(self, msg_lst):
    """
    Sign a message with a secure signature.
    """
    h = self._auth.copy()
    for m in msg_lst:
      h.update(m)
    return self._str_to_bytes(h.hexdigest())

  def serialize_wire_msg(self, msg_type, content=None, parent_header=None, metadata=None, identities=None):
    header = self._new_header(msg_type)
    if content is None:
      content = {}
    if parent_header is None:
      parent_header = {}
    if metadata is None:
      metadata = {}

    def encode(msg):
      return self._str_to_bytes(json.dumps(msg))

    msg_lst = [
      encode(header),
      encode(parent_header),
      encode(metadata),
      encode(content),
    ]
    signature = self.sign(msg_lst)
    parts = [DELIM,
         signature,
         msg_lst[0],
         msg_lst[1],
         msg_lst[2],
         msg_lst[3]]
    if identities:
      parts = identities + parts

    return parts

  def deserialize_wire_msg(self, wire_msg):
    """split the routing prefix and message frames from a message on the wire"""
    delim_idx = wire_msg.index(DELIM)
    identities = wire_msg[:delim_idx]
    m_signature = wire_msg[delim_idx + 1]
    msg_frames = wire_msg[delim_idx + 2:]

    def decode(msg):
      dprint(1, "decode", msg)
      return json.loads(msg.decode('ascii') if PYTHON3 else msg)

    m = {}
    m['header']        = decode(msg_frames[0])
    m['parent_header'] = decode(msg_frames[1])
    m['metadata']      = decode(msg_frames[2])
    m['content']       = decode(msg_frames[3])
    dprint(1, "will sign", m)
    check_sig = self.sign(msg_frames)
    if check_sig != m_signature:
      dprint(1, check_sig ,"!=", m_signature)
      raise ValueError("Signatures do not match")

    dprint(1, "m", m)
    dprint(1, "identities", identities)
    return identities, m

class OutgoingStream:
  def __init__(self, wire, stream):
    self._wire = wire
    self._stream = stream

  def send(self, msg_type, content=None, parent_header=None, metadata=None, identities=None):
    parts = self._wire.serialize_wire_msg(msg_type, content=content, parent_header=parent_header, metadata=metadata, identities=identities)
    dprint(3, "send parts:", parts)
    self._stream.send_multipart(parts)
    self._stream.flush()

class ShellHandler:
  def __init__(self, engine_id, iopub, shell, driver_info, driver):
    self._driver_info = driver_info
    self._driver = driver
    self._engine_id = engine_id
    self._iopub = iopub
    self._shell = shell
    self._execution_count = 1
    self._pending_execute_requests = []
    self._pending_execute_request = False

  def _begin(self, identities, msg, on_done):
    execution_count = self._execution_count
    started = datetime.datetime.now().isoformat()
    parent_header = msg['header']
    code = msg['content']["code"]

    self._iopub.send('status', {'execution_state': "busy"}, parent_header=parent_header)

    self._execution_count += 1
    content = {
      'execution_count': execution_count,
      'code': code,
    }
    self._iopub.send('execute_input', content, parent_header=parent_header)

    def _done(result_data, result_metadata=None):
      if result_metadata is None:
        result_metadata = {}

      self._iopub.send('status', {'execution_state': "idle"}, parent_header=parent_header)

      content = {
        'execution_count': execution_count,
        'data': result_data,
        'metadata': result_metadata
      }
      self._iopub.send('execute_result', content, parent_header=parent_header)

      metadata = {
        "dependencies_met": True,
        "engine": self._engine_id,
        "status": "ok",
        "started": started,
      }
      content = {
        "status": "ok",
        "execution_count": execution_count,
        "user_variables": {},
        "payload": [],
        "user_expressions": {},
      }
      self._shell.send('execute_reply', content, metadata=metadata,
        parent_header=parent_header, identities=identities)

      on_done()

    return _done

  def execute_request(self, identities, msg):
    def schedule_next():
      print("schedule_next", self._pending_execute_request, self._pending_execute_requests)

      if len(self._pending_execute_requests) == 0:
        self._pending_execute_request = False
      else:
        identities2, msg2 = self._pending_execute_requests.pop(0)
        self._execute_request(schedule_next, identities2, msg2)

    if self._pending_execute_request:
      self._pending_execute_requests.append((identities, msg))
    else:
      self._execute_request(schedule_next, identities, msg)


  def _execute_request(self, on_done, identities, msg):
    on_result = self._begin(identities, msg, on_done)

    code = msg['content']["code"]

    has_displayed = set()
    def on_display(display_id, data, metadata):
      content = {
        "data": data,
        "metadata": metadata,
        "transient": {
          "display_id": display_id,
        },
      }

      display_message_type = 'update_display_data'
      if display_id not in has_displayed:
        display_message_type = 'display_data'
        has_displayed.add(display_id)
      self._iopub.send(display_message_type, content, parent_header=msg['header'])

    def on_stdout(text):
      content = {
        'name': "stdout",
        'text': text,
      }
      self._iopub.send('stream', content, parent_header=msg['header'])

    self._driver(code, on_stdout, on_display, on_result)

  def kernel_info_request(self, identities, msg):
    content = {}
    content.update(self._driver_info)
    content.update({
      "protocol_version": "5.0",
      "ipython_version": [1, 1, 0, ""],
    })
    self._shell.send('kernel_info_reply', content, parent_header=msg['header'], identities=identities)

  def __call__(self, identities, msg):
    dprint(1, "shell received:", identities, msg)

    # process request:
    msg_type = msg['header']["msg_type"]
    if msg_type == "execute_request":
      self.execute_request(identities, msg)
    elif msg_type == "kernel_info_request":
      self.kernel_info_request(identities, msg)
    elif msg_type == "history_request":
      dprint(1, "unhandled history request")
    else:
      dprint(1, "unknown msg_type:", msg_type)

class Kernel:
  def __init__(self, config, driver_info, driver):
    # Clone config so we can update it.
    config = json.loads(json.dumps(config))

    self._config = config
    self._exiting = False
    self._engine_id = str(uuid.uuid4())
    self._wire = WireProtocol(self._engine_id, config["key"], config["signature_scheme"])

    connection = config["transport"] + "://" + config["ip"]

    def bind(socket, port):
      if port <= 0:
        return socket.bind_to_random_port(connection)
      else:
        socket.bind("%s:%s" % (connection, port))
      return port

    def wrap_with_deserialization(fn):
      def accept(wire_msg):
        return fn(*self._wire.deserialize_wire_msg(wire_msg))

      return accept

    ## Initialize:
    ioloop.install()

    ctx = zmq.Context()
    self._heartbeat_socket = ctx.socket(zmq.REP)
    config["hb_port"] = bind(self._heartbeat_socket, config["hb_port"])

    # IOPub/Sub: also called SubSocketChannel in IPython sources
    self._iopub_socket = ctx.socket(zmq.PUB)
    config["iopub_port"] = bind(self._iopub_socket, config["iopub_port"])
    iopub_stream = zmqstream.ZMQStream(self._iopub_socket)
    iopub_stream.on_recv(wrap_with_deserialization(self._iopub_handler))
    iopub = OutgoingStream(self._wire, iopub_stream)

    self._control_socket = ctx.socket(zmq.ROUTER)
    config["control_port"] = bind(self._control_socket, config["control_port"])
    control_stream = zmqstream.ZMQStream(self._control_socket)
    control_stream.on_recv(wrap_with_deserialization(self._control_handler))

    self._stdin_socket = ctx.socket(zmq.ROUTER)
    config["stdin_port"] = bind(self._stdin_socket, config["stdin_port"])
    stdin_stream = zmqstream.ZMQStream(self._stdin_socket)
    stdin_stream.on_recv(wrap_with_deserialization(self._stdin_handler))

    self._shell_socket = ctx.socket(zmq.ROUTER)
    config["shell_port"] = bind(self._shell_socket, config["shell_port"])
    shell_stream = zmqstream.ZMQStream(self._shell_socket)
    shell = OutgoingStream(self._wire, shell_stream)
    shell_stream.on_recv(wrap_with_deserialization(self._shell_handler))

    self._shell_handler_impl = ShellHandler(self._engine_id, iopub, shell, driver_info, driver)

  def _control_handler(self, identities, msg):
    dprint(1, "control received:", identities, msg)
    msg_type = msg['header']["msg_type"]
    if msg_type == "shutdown_request":
      self._shutdown_request(identities, msg)

  def _shell_handler(self, identities, msg):
    msg_type = msg['header']["msg_type"]
    if msg_type == "shutdown_request":
      self._shutdown_request(identities, msg)
    else:
      self._shell_handler_impl(identities, msg)

  def _shutdown_request(self, identities, msg):
    self.shutdown()

  def _iopub_handler(self, identities, msg):
    dprint(1, "iopub received:", identities, msg)

  def _stdin_handler(self, identities, msg):
    dprint(1, "stdin received:", identities, msg)

  # Utility functions:
  def shutdown(self):
    self._exiting = True
    ioloop.IOLoop.instance().stop()

  def run(self):
    dprint(1, "Config:", json.dumps(self._config))
    dprint(1, "Starting loops...")

    def heartbeat_loop():
      dprint(2, "Starting loop for 'Heartbeat'...")
      while not self._exiting:
        dprint(3, ".", end="")
        try:
          zmq.device(zmq.FORWARDER, self._heartbeat_socket, self._heartbeat_socket)
        except zmq.ZMQError as e:
          if e.errno == errno.EINTR:
            continue
          else:
            raise
        else:
          break
    hb_thread = threading.Thread(target=heartbeat_loop)
    hb_thread.daemon = True
    hb_thread.start()

    dprint(1, "Ready! Listening...")

    ioloop.IOLoop.instance().start()
