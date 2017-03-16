from os import path

from py_mini_racer import py_mini_racer

class PalletParser:
  def __init__(self, src_root, pkg_root):
    self._pallet = []
    self._source_cache = {}
    self._import_cache = {}

    self._attempts = [
      {
        "language": "nao",
        "suffix": ".nao",
        "dir": src_root,
      },
      {
        "language": "python",
        "suffix": ".py",
        "dir": src_root,
      },
      {
        "language": "tensorflow:metagraph:pbtxt",
        "suffix": ".metagraph.pbtxt",
        "dir": pkg_root,
      },
    ]

  def pallet(self):
    return self._pallet[:]

  def put_source(self, import_path, source):
    self._source_cache[import_path] = source

  def _enumerate_imports(self, package_expr):
    if package_expr[0] != "_sf_package":
      raise Exception("Not a package expression: " + package_expr)

    name, *exprs = package_expr[1:]
    imported = []
    for expr in exprs:
      if expr[0] != "_sf_import":
        continue

      for import_name, import_path, imported_scope in expr[1]:
        imported.append((import_path, imported_scope))

    return imported

  # TODO(adamb) Modify to *only* take input via stdin/command-line.
  def _parse_external(self, source):
    ctx = py_mini_racer.MiniRacer()
    with open(path.join(path.dirname(__file__), "../lib/parse.js")) as f:
      ctx.eval(f.read())
    expr = ctx.call("parse.parseExpressions", source)
    # pp(expr)
    return expr

  def _attempt(self, language, import_path, imported_scope, filepath=None, content=None):
    return {
      "language": language,
      "import_path": import_path,
      "imported_package_name": path.basename(import_path),
      "imported_scope_name": imported_scope,
      "path": filepath,
      "content": content,
    }

  def _resolve_import(self, import_path, imported_scope):
    if import_path in self._source_cache:
      source = self._source_cache[import_path]
      return self._attempt("nao", import_path, imported_scope, content=source)

    for attempt in self._attempts:
      filepath = path.join(attempt["dir"], import_path + attempt["suffix"])
      if path.isfile(filepath):
        return self._attempt(
            attempt["language"],
            import_path,
            imported_scope,
            filepath=filepath)

    raise Exception("No such import path: " + import_path)

  def resolve_import(self, import_path, imported_scope):
    cache_key = (import_path, imported_scope)
    if cache_key in self._import_cache:
      return self._import_cache[cache_key]

    resolved = self._resolve_import(import_path, imported_scope)
    if resolved["language"] == "nao":
      source = resolved["content"]
      if source is None:
        with open(resolved["path"]) as f:
          source = f.read()
      exprs = self._parse_external(source)

      pkg = ["_sf_package", resolved["imported_package_name"], *exprs]
      for import_path, imported_scope in self._enumerate_imports(pkg):
        # Skip imports that provide direct access to TensorFlow internals.
        if import_path == "tensorflow":
          continue

        self.resolve_import(import_path, imported_scope)
    else:
      pkg = ["_sf_foreign_package", resolved["language"], resolved["imported_package_name"], resolved["imported_scope_name"], resolved["path"]]

    self._import_cache[cache_key] = pkg
    self._pallet.append(pkg)
    return pkg
