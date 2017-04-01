import imp
import inspect
import json
import tensorflow as tf

from tensorflow.python.ops import script_ops

from nao.structure import graph_ffi
from nao.compiler.primitive_function import PrimitiveFunction
from nao.compiler.python_package import PythonPackage

class ImportedPythonFunction:
  def __init__(self, fn):
    self._fn = fn
    sig = inspect.signature(fn)
    self._Tout = sig.return_annotation
    self._argnames = sig.parameters.keys()
    self.__name__ = self._fn.__name__

  def __call__(self, *args):
    return tf.py_func(
        func=self._fn,
        inp=args,
        Tout=self._Tout,
        stateful=True, # TODO
        name=None)


_python_importer = graph_ffi.PythonImporter()

def finish():
  py_func_data = _python_importer.dump_py_funcs(script_ops._py_funcs)
  tf.constant(json.dumps(py_func_data), name="py_funcs_json")

def make_compile_fn(workspace, import_path, tags):
  source = workspace.read_src(import_path + ".py")
  if source is None:
    return None

  def compile(resolved_imports, previous):
    outer_module = imp.new_module("%s$wrapper" % import_path)
    all_fns = _python_importer.import_module(import_path, source)
    for fn_name, fn in all_fns.items():
      imported_py_func = ImportedPythonFunction(fn)
      setattr(outer_module, fn_name, imported_py_func)

    return PythonPackage(outer_module)

  # TODO(adamb) Considering returning all the above logic as "imports",
  #     so we can hide use of _python_importer
  return ([], compile)
