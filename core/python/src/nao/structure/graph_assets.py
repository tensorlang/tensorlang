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
