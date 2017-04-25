
_summary_writer = None

class Multiplexer:
  def __init__(self, targets):
    self._targets = targets

  def add_target(self, target):
    self._targets.append(target)

  def remove_target(self, target):
    self._targets.remove(target)

  def add_event(self, event):
    for target in self._targets:
      target.add_event(event)

  def flush(self):
    for target in self._targets:
      target.flush()

  def add_graph(self, graph, global_step=None):
    for target in self._targets:
      target.add_graph(graph, global_step)

  def add_meta_graph(self, meta_graph_def, global_step=None):
    for target in self._targets:
      target.add_meta_graph(meta_graph_def, global_step)

  def add_run_metadata(self, run_metadata, tag, global_step=None):
    for target in self._targets:
      target.add_run_metadata(run_metadata, tag, global_step)

  def add_summary(self, summary, global_step=None):
    for target in self._targets:
      target.add_summary(summary, global_step)

  def add_session_log(self, session_log, global_step=None):
    for target in self._targets:
      target.add_session_log(session_log, global_step)

class Delegate:
  def __init__(self, fn):
    self._fn = fn

  def flush(self):
    pass

  def add_event(self, event):
    self._fn(["event", event])

  def add_graph(self, graph, global_step=None):
    self._fn(["graph", graph, global_step])

  def add_meta_graph(self, meta_graph_def, global_step=None):
    self._fn(["meta_graph", meta_graph_def, global_step])

  def add_run_metadata(self, run_metadata, tag, global_step=None):
    self._fn(["run_metadata", run_metadata, tag, global_step])

  def add_summary(self, summary, global_step=None):
    self._fn(["summary", summary, global_step])

  def add_session_log(self, session_log, global_step=None):
    self._fn(["session_log", session_log, global_step])

def set_summary_writer(summary_writer):
  global _summary_writer
  _summary_writer = summary_writer

def get_summary_writer():
  global _summary_writer
  return _summary_writer
