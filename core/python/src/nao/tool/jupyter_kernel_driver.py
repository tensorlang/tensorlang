class Driver:
  def __init__(self, repl_session):
    self._repl_session = repl_session

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

  def do(self, code, on_stdout, on_result):
    try:
      result = self._repl_session.run(code)
    except Exception as e:
      result = e
    on_result({"text/plain": str(result)})
