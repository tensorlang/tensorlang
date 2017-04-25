import ast
import imp
import sys

def eprint(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

class PythonImporter:
  def __init__(self):
    self._module_sources = {} # module_name -> module_source
    self._imported_functions = {} # fn -> (module_name, fn_name)

  def import_module(self, module_name, module_src):
    # Documentation here http://greentreesnakes.readthedocs.io/en/latest/nodes.html
    inner_module_ast = ast.parse(module_src)
    fn_asts = [f for f in inner_module_ast.body if isinstance(f, ast.FunctionDef)]

    fn_names = [fn_ast.name for fn_ast in fn_asts]
    return self._load_module_functions(module_name, module_src, fn_names)

  def dump_py_funcs(self, py_funcs):
    with py_funcs._lock:
      return {
        'unique_id': py_funcs._unique_id,
        'modules': self._dump_modules(py_funcs._funcs)
      }

  def restore_py_funcs(self, py_funcs, data):
    unique_id = data['unique_id']
    modules = data['modules']

    # eprint('py_funcs._funcs', py_funcs._funcs)
    with py_funcs._lock:
      if len(py_funcs._funcs) > 0:
        raise Exception(
            "py_funcs is not pristine (len(py_funcs._funcs) is %d). Aborting restore." % len(py_funcs._funcs))

      py_funcs._unique_id = unique_id
      py_funcs._funcs = self._load_funcs(modules)

  def _dump_modules(self, fn_by_token_dict):
    modules = {}
    for token, fn in fn_by_token_dict.items():
      if fn not in self._imported_functions:
        continue
      module_name, fn_name = self._imported_functions[fn]
      # eprint("dumping %s -> (%s, %s)" % (fn, module_name, fn_name))
      if module_name not in modules:
        modules[module_name] = {
          "source": self._module_sources[module_name],
          "fn_name_by_token": {},
        }
      modules[module_name]["fn_name_by_token"][token] = fn_name

    return modules

  def _load_funcs(self, modules):
    fn_by_token = {}
    for module_name, data in modules.items():
      source, fn_name_by_token = data['source'], data['fn_name_by_token']
      fn_names = []
      for token, fn_name in fn_name_by_token.items():
        fn_names.append(fn_name)

      fns = self._load_module_functions(module_name, source, fn_names)
      for token, fn_name in fn_name_by_token.items():
        fn_by_token[token] = fns[fn_name]

    return fn_by_token

  def _load_module_functions(self, name, source, fn_names):
    m = imp.new_module(name)
    exec(source, m.__dict__)

    d = {}
    for fn_name in fn_names:
      fn = getattr(m, fn_name)
      self._imported_functions[fn] = (name, fn_name)
      d[fn_name] = fn

    self._module_sources[name] = source

    # eprint('_load_module_functions', d)
    return d
