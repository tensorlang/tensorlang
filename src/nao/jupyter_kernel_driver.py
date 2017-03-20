class Driver:
  def __init__(self, repl_session):
    self._repl_session = repl_session

  def info(self):
    return {
      "language_version": [0, 0, 1],
      "language": "simple_kernel",
      "implementation": "simple_kernel",
      "implementation_version": "1.1",
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
    result = self._repl_session.run(code)
    # on_stdout(code)
    on_result({"text/plain": str(result)})
