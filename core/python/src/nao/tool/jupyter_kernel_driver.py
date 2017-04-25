import base64
import imghdr
import json
import queue
import sys
import threading
import time

from nao.tool import summary_format, json_util

_IMGHDR_TO_MIMETYPE = {
    'bmp': 'image/bmp',
    'gif': 'image/gif',
    'jpeg': 'image/jpeg',
    'png': 'image/png'
}
_DEFAULT_IMAGE_MIMETYPE = 'application/octet-stream'

def _content_type_for_image(encoded_image_string):
  image_type = imghdr.what(None, encoded_image_string)
  return _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)

class Driver:
  def __init__(self, repl_session):
    self._repl_session = repl_session
    self._id = 0
    self._display_queue = queue.Queue()
    self._display_thread = None

  def info(self):
    return {
      "language_version": [0, 0, 1],
      "language": "nao",
      "implementation": "nao",
      "implementation_version": "0.0.1",
      "language_info": {
        "name": "nao",
        "version": "1.0",
        'mimetype': "",
        'file_extension': ".nao",
        'pygments_lexer': "",
        'codemirror_mode': "",
        'nbconvert_exporter': "",
      },
      "banner": ""
    }

  def _emit_summary_pb(self, on_display, protobuf, wall_time, step):
    def display_summary(summary_type, summary):
      self._id = self._id + 1

      data = {}
      metadata = {}
      if summary_type == "image":
        encoded_image_string = summary["encoded_image_string"]
        content_type = _content_type_for_image(encoded_image_string)
        data[content_type] = base64.b64encode(encoded_image_string).decode('utf-8')
        metadata[content_type] = {
          "width": summary["width"],
          "height": summary["height"],
        }
      else:
        content_type = "application/json"
        data[content_type] = json_util.Cleanse(summary)

      on_display(str(self._id), data, metadata)

    summary_format.parse(protobuf, wall_time, int(step), display_summary)

  def _run_display_thread(self):
    q = self._display_queue
    while True:
      display_data = q.get()
      try:
        if display_data is None:
          break
        self._emit_summary_pb(*display_data)
      except Exception as e:
        print("ERROR in _run_display_thread", e, file=sys.stderr)
      finally:
        q.task_done()

  def start(self):
    if self._display_thread:
      return

    self._display_thread = threading.Thread(target=self._run_display_thread)
    self._display_thread.start()

  def do(self, code, on_stdout, on_display, on_result):
    self.start()

    q = self._display_queue
    def on_summary_protobuf(args):
      if args[0] is not "summary":
        return

      protobuf, step = args[1], args[2]
      q.put((on_display, protobuf, time.time(), step))

    try:
      result = self._repl_session.run(code, summary_fn=on_summary_protobuf)
    except Exception as e:
      result = e
    on_result({"text/plain": str(result)})
