import tensorflow as tf

from nao.structure import graph_assets

_ASSETS = []

def finish():
  asset_map = graph_assets.consolidate_to_asset_map(_ASSETS)
  graph_assets.store_asset_map(asset_map)

def make_compile_fn(workspace, import_path, tags):
  if not tags.get("asset", False):
    return None

  url = tags["url"],
  sha256 = tags.get("sha256", None),

  def compile(resolved_imports, previous):
    placeholder_name = import_path.split("/")[-1].replace("-", "_").replace(".", "_")
    placeholder = tf.placeholder(tf.string, tf.TensorShape([]), placeholder_name)
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
