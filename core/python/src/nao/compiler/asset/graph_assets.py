import tensorflow as tf

from six.moves import urllib

from os import path
from tensorflow.python.platform import gfile

from nao.compiler.asset.retry import retry
from nao.structure import graph_constants

_ASSET_MAP_JSON_KEY = "asset_map_json"

# Returns {"asset_name": {"url", "digest"}}
def load_asset_map(graph):
  asset_map = graph_constants.load_json(graph, _ASSET_MAP_JSON_KEY)
  if asset_map is None:
    return {}
  return asset_map

# asset_map = {"asset_name": {"url", "digest"}}
def store_asset_map(asset_map):
  return graph_constants.store_json(_ASSET_MAP_JSON_KEY, asset_map)

# assets = [{"name", "digest", "url"}]
# Raise if two assets have the same name but different urls or different digests
def consolidate_to_asset_map(assets):
  result = {} # "name" -> asset
  collisions = {} # {"name" -> [asset]}
  for asset in assets:
    name = asset["name"]
    if name in result:
      collisions[name] = [result[name]]
      del result[name]
    elif name in collisions:
      collisions[name].append(asset)
    else:
      result[name] = asset

  if len(collisions) > 0:
    raise Exception("Found asset collisions: %s" % collisions)

  return result


_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
  return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS

@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def _urlretrieve_with_retry(url, filename=None):
  return urllib.request.urlretrieve(url, filename)

def maybe_download(filepath, source_url):
  """Download the data from source url, unless it's already here.
  Args:
      basename: string, name of the file in the directory.
      target_dir: string, path to working directory.
      source_url: url to download from if file doesn't exist.
  Returns:
      Path to resulting file.
  """
  target_dir = path.dirname(filepath)
  if not gfile.Exists(target_dir):
    gfile.MakeDirs(target_dir)
  if not gfile.Exists(filepath):
    print('Downloading', source_url, 'to', filepath)
    temp_file_name, _ = _urlretrieve_with_retry(source_url)
    gfile.Copy(temp_file_name, filepath)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filepath, size, 'bytes.')
  return filepath
