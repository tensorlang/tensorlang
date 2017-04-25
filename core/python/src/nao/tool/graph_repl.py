import traceback
import sys
import tensorflow as tf

from nao.run import graph_summary

from nao.compiler.retvalbag import RetvalBag

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class ReplSession:
  def __init__(self, compiler, log_dir_fn):
    self._log_dir_fn = log_dir_fn
    self._suffix = ".nao"
    self._compiler = compiler
    self._session = compiler.new_session()
    self._graph = self._session.graph
    self._previous_queue_runners = frozenset()
    self._previous_vars = frozenset()
    self._threads = []
    self._coord = tf.train.Coordinator()
    self._next_run_id = 0
    self._summary_writer = None

  def _vars(self):
    with self._graph.as_default():
      return tf.global_variables()

  def _queue_runners(self):
    with self._graph.as_default():
      return tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)

  def _init_new_vars(self, new_vars):
    if len(new_vars) == 0:
      return

    print("New variables", new_vars)
    tf.variables_initializer(new_vars).run(session=self._session)

  def _init_new_queue_runners(self, new_queue_runners):
    if len(new_queue_runners) == 0:
      return

    print("new_queue_runners", new_queue_runners)
    for qr in new_queue_runners:
      threads = qr.create_threads(sess, coord=self._coord, daemon=True, start=True)
      print("started", threads)
      self._threads.extend(threads)

  def run(self, src, summary_fn=None):
    run_id = self._next_run_id
    self._next_run_id = self._next_run_id + 1
    if self._summary_writer is None:
      self._summary_writer = tf.summary.FileWriter(
          self._log_dir_fn())

    try:
      multiplexer = graph_summary.Multiplexer([self._summary_writer])
      if summary_fn is not None:
        multiplexer.add_target(graph_summary.Delegate(summary_fn))

      graph_summary.set_summary_writer(multiplexer)
      return self._run(multiplexer, run_id, src)
    finally:
      graph_summary.set_summary_writer(None)
      self._summary_writer.flush()

  def _run(self, summary_writer, run_id, src):
    self._compiler.clear()
    self._compiler.put_source("main%s" % self._suffix, src)
    self._compiler.set_default_device("/cpu:0")

    above = None
    pkg = self._compiler.resolve_import_path("main", reimport=True)
    above = pkg.ctx().get_above()

    # Write graph once we've generated it.
    summary_writer.add_graph(self._session.graph, run_id)
    summary_writer.flush()

    vars = frozenset(self._vars())
    self._init_new_vars(vars - self._previous_vars)
    self._previous_vars = vars

    queue_runners = frozenset(self._queue_runners())
    self._init_new_queue_runners(queue_runners - self._previous_queue_runners)
    self._previous_queue_runners = queue_runners

    if isinstance(above, RetvalBag):
      above = above.get(None)

    if isinstance(above, (tf.Tensor, tf.Variable, tf.Operation)):
      run_metadata = tf.RunMetadata()
      above = self._session.run(above, run_metadata=run_metadata)
      summary_writer.add_run_metadata(run_metadata, "repl-%04d" % run_id, run_id)

    return above

  def __del__(self):
    # Shutdown threads, if any.
    if self._summary_writer is not None:
      self._summary_writer.close()
    self._coord.request_stop()
    self._coord.join(self._threads)

    self._session.close()

import atexit
import os
import readline

HISTORY_BASENAME = '.nao_history'

def run(parser, log_fn):
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

  repl_session = ReplSession(parser, log_fn)
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
      print(result)
    except Exception as e:
      print("".join(traceback.format_exception(None, e, e.__traceback__)),
            file=sys.stdout, flush=True)


  #
  #
  # """A training helper that checkpoints models and computes summaries.
  #
  # The Supervisor is a small wrapper around a `Coordinator`, a `Saver`,
  # and a `SessionManager` that takes care of common needs of TensorFlow
  # training programs.
  #
  # #### Use for a single program
  #
  # ```python
  # with tf.Graph().as_default():
  #   ...add operations to the graph...
  #   # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
  #   sv = Supervisor(logdir='/tmp/mydir')
  #   # Get a TensorFlow session managed by the supervisor.
  #   with sv.managed_session(FLAGS.master) as sess:
  #     # Use the session to train the graph.
  #     while not sv.should_stop():
  #       sess.run(<my_train_op>)
  # ```
  #
  # Within the `with sv.managed_session()` block all variables in the graph have
  # been initialized.  In addition, a few services have been started to
  # checkpoint the model and add summaries to the event log.
  #
  # If the program crashes and is restarted, the managed session automatically
  # reinitialize variables from the most recent checkpoint.
  #
  # The supervisor is notified of any exception raised by one of the services.
  # After an exception is raised, `should_stop()` returns `True`.  In that case
  # the training loop should also stop.  This is why the training loop has to
  # check for `sv.should_stop()`.
  #
  # Exceptions that indicate that the training inputs have been exhausted,
  # `tf.errors.OutOfRangeError`, also cause `sv.should_stop()` to return `True`
  # but are not re-raised from the `with` block: they indicate a normal
  # termination.
  #
  # #### What `master` string to use
  #
  # Whether you are running on your machine or in the cluster you can use the
  # following values for the --master flag:
  #
  # * Specifying `''` requests an in-process session that does not use RPC.
  #
  # * Specifying `'local'` requests a session that uses the RPC-based
  #   "Master interface" to run TensorFlow programs. See
  #   [`tf.train.Server.create_local_server()`](#Server.create_local_server) for
  #   details.
  #
  # #### Advanced use
  #
  # ##### Launching additional services
  #
  # `managed_session()` launches the Checkpoint and Summary services (threads).
  # If you need more services to run you can simply launch them in the block
  # controlled by `managed_session()`.
  #
  # Example: Start a thread to print losses.  We want this thread to run
  # every 60 seconds, so we launch it with `sv.loop()`.
  #
  #   ```python
  #   ...
  #   sv = Supervisor(logdir='/tmp/mydir')
  #   with sv.managed_session(FLAGS.master) as sess:
  #     sv.loop(60, print_loss, (sess, ))
  #     while not sv.should_stop():
  #       sess.run(my_train_op)
  #   ```
  #
  # ##### Launching fewer services
  #
  # `managed_session()` launches the "summary" and "checkpoint" threads which use
  # either the optionally `summary_op` and `saver` passed to the constructor, or
  # default ones created automatically by the supervisor.  If you want to run
  # your own summary and checkpointing logic, disable these services by passing
  # `None` to the `summary_op` and `saver` parameters.
  #
  # Example: Create summaries manually every 100 steps in the chief.
  #
  #   ```python
  #   # Create a Supervisor with no automatic summaries.
  #   sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  #   # As summary_op was None, managed_session() does not start the
  #   # summary thread.
  #   with sv.managed_session(FLAGS.master) as sess:
  #     for step in xrange(1000000):
  #       if sv.should_stop():
  #         break
  #       if is_chief and step % 100 == 0:
  #         # Create the summary every 100 chief steps.
  #         sv.summary_computed(sess, sess.run(my_summary_op))
  #       else:
  #         # Train normally
  #         sess.run(my_train_op)
  #   ```
  #
  # ##### Custom model initialization
  #
  # `managed_session()` only supports initializing the model by running an
  # `init_op` or restoring from the latest checkpoint.  If you have special
  # initialization needs, see how to specify a `local_init_op` when creating the
  # supervisor.  You can also use the `SessionManager` directly to create a
  # session and check if it could be initialized automatically.
  #
  #   """Create a `Supervisor`.
  #
  #   Args:
  #     graph: A `Graph`.  The graph that the model will use.  Defaults to the
  #       default `Graph`.  The supervisor may add operations to the graph before
  #       creating a session, but the graph should not be modified by the caller
  #       after passing it to the supervisor.
  #     ready_op: 1-D string `Tensor`.  This tensor is evaluated by supervisors in
  #       `prepare_or_wait_for_session()` to check if the model is ready to use.
  #       The model is considered ready if it returns an empty array.  Defaults to
  #       the tensor returned from `tf.report_uninitialized_variables()`  If
  #       `None`, the model is not checked for readiness.
  #     ready_for_local_init_op: 1-D string `Tensor`.  This tensor is evaluated by
  #       supervisors in `prepare_or_wait_for_session()` to check if the model is
  #       ready to run the local_init_op.
  #       The model is considered ready if it returns an empty array.  Defaults to
  #       the tensor returned from
  #       `tf.report_uninitialized_variables(tf.global_variables())`. If `None`,
  #       the model is not checked for readiness before running local_init_op.
  #     is_chief: If True, create a chief supervisor in charge of initializing
  #       and restoring the model.  If False, create a supervisor that relies
  #       on a chief supervisor for inits and restore.
  #     init_op: `Operation`.  Used by chief supervisors to initialize the model
  #       when it can not be recovered.  Defaults to an `Operation` that
  #       initializes all variables.  If `None`, no initialization is done
  #       automatically unless you pass a value for `init_fn`, see below.
  #     init_feed_dict: A dictionary that maps `Tensor` objects to feed values.
  #       This feed dictionary will be used when `init_op` is evaluated.
  #     local_init_op: `Operation`. Used by all supervisors to run initializations
  #       that should run for every new supervisor instance. By default these
  #       are table initializers and initializers for local variables.
  #       If `None`, no further per supervisor-instance initialization is
  #       done automatically.
  #     logdir: A string.  Optional path to a directory where to checkpoint the
  #       model and log events for the visualizer.  Used by chief supervisors.
  #       The directory will be created if it does not exist.
  #     summary_op: An `Operation` that returns a Summary for the event logs.
  #       Used by chief supervisors if a `logdir` was specified.  Defaults to the
  #       operation returned from summary.merge_all().  If `None`, summaries are
  #       not computed automatically.
  #     saver: A Saver object.  Used by chief supervisors if a `logdir` was
  #       specified.  Defaults to the saved returned by Saver().
  #       If `None`, the model is not saved automatically.
  #     global_step: An integer Tensor of size 1 that counts steps.  The value
  #       from 'global_step' is used in summaries and checkpoint filenames.
  #       Default to the op named 'global_step' in the graph if it exists, is of
  #       rank 1, size 1, and of type tf.int32 or tf.int64.  If `None` the global
  #       step is not recorded in summaries and checkpoint files.  Used by chief
  #       supervisors if a `logdir` was specified.
  #     save_summaries_secs: Number of seconds between the computation of
  #       summaries for the event log.  Defaults to 120 seconds.  Pass 0 to
  #       disable summaries.
  #     save_model_secs: Number of seconds between the creation of model
  #       checkpoints.  Defaults to 600 seconds.  Pass 0 to disable checkpoints.
  #     recovery_wait_secs: Number of seconds between checks that the model
  #       is ready.  Used by supervisors when waiting for a chief supervisor
  #       to initialize or restore the model.  Defaults to 30 seconds.
  #     stop_grace_secs: Grace period, in seconds, given to running threads to
  #       stop when `stop()` is called.  Defaults to 120 seconds.
  #     checkpoint_basename: The basename for checkpoint saving.
  #     session_manager: `SessionManager`, which manages Session creation and
  #       recovery. If it is `None`, a default `SessionManager` will be created
  #       with the set of arguments passed in for backwards compatibility.
  #     summary_writer: `SummaryWriter` to use or `USE_DEFAULT`.  Can be `None`
  #       to indicate that no summaries should be written.
  #     init_fn: Optional callable used to initialize the model. Called
  #       after the optional `init_op` is called.  The callable must accept one
  #       argument, the session being initialized.
  #
  #   Returns:
  #     A `Supervisor`.
  #   """
