import tensorflow as tf

from nao.compiler.asset import graph_assets

_ASSETS = []

def finish():
  asset_map = graph_assets.consolidate_to_asset_map(_ASSETS)
  graph_assets.store_asset_map(asset_map)

def asset_map():
  return graph_assets.consolidate_to_asset_map(_ASSETS)

def make_compile_fn(workspace, import_path, tags):
  if not tags.get("asset", False):
    return None

  url = tags["url"]
  sha256 = tags.get("sha256", None)
  asset_path = workspace.find_asset_path(import_path)

  def maybe_download(path, url):
    print("maybe_download", "path", path, type(path), "url", url, type(url))

    return graph_assets.maybe_download(
        path.decode('utf-8'),
        url.decode('utf-8')).encode('utf-8')

  def compile(resolved_imports, previous):
    placeholder_name = import_path.split("/")[-1].replace("-", "_").replace(".", "_")
    # placeholder = tf.placeholder(tf.string, tf.TensorShape([]), placeholder_name)
    placeholder = tf.py_func(
        maybe_download,
        [asset_path, url],
        tf.string,
        name=placeholder_name)

    _ASSETS.append(
      {
        "placeholder": placeholder.name,
        "name": import_path,
        "url": url,
        "sha256": sha256,
      }
    )
    return placeholder

  return ([], compile)
