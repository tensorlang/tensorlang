# Taken from https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py

from collections import namedtuple

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2

# The tag that values containing health pills have. Health pill data is stored
# in tensors. In order to distinguish health pill values from scalar values, we
# rely on how health pill values have this special tag value.
HEALTH_PILL_EVENT_TAG = '__health_pill__'

def _CompressHistogram(histo_ev, bps):
  """Creates fixed size histogram by adding compression to accumulated state.

  This routine transforms a histogram at a particular step by linearly
  interpolating its variable number of buckets to represent their cumulative
  weight at a constant number of compression points. This significantly reduces
  the size of the histogram and makes it suitable for a two-dimensional area
  plot where the output of this routine constitutes the ranges for a single x
  coordinate.

  Args:
    histo_ev: A HistogramEvent namedtuple.
    bps: Compression points represented in basis points, 1/100ths of a percent.

  Returns:
    CompressedHistogramEvent namedtuple.
  """
  # See also: Histogram::Percentile() in core/lib/histogram/histogram.cc
  histo = histo_ev.histogram_value
  if not histo.num:
    return {
      "wall_time": histo_ev.wall_time,
      "step": histo_ev.step,
      "compressed_histogram_values": [{"basis_point": b, "value": 0.0} for b in bps]
    }
  bucket = np.array(histo.bucket)
  weights = (bucket * bps[-1] / (bucket.sum() or 1.0)).cumsum()
  values = []
  j = 0
  while j < len(bps):
    i = np.searchsorted(weights, bps[j], side='right')
    while i < len(weights):
      cumsum = weights[i]
      cumsum_prev = weights[i - 1] if i > 0 else 0.0
      if cumsum == cumsum_prev:  # prevent remap divide by zero
        i += 1
        continue
      if not i or not cumsum_prev:
        lhs = histo.min
      else:
        lhs = max(histo.bucket_limit[i - 1], histo.min)
      rhs = min(histo.bucket_limit[i], histo.max)
      weight = _Remap(bps[j], cumsum_prev, cumsum, lhs, rhs)
      values.append({"basis_point": bps[j], "value": weight})
      j += 1
      break
    else:
      break
  while j < len(bps):
    values.append({"basis_point": bps[j], "value": histo.max})
    j += 1
  return {"wall_time": histo_ev.wall_time, "step": histo_ev.step, "compressed_histogram_values": values}


def _Remap(x, x0, x1, y0, y1):
  """Linearly map from [x0, x1] unto [y0, y1]."""
  return y0 + (x - x0) * float(y1 - y0) / (x1 - x0)

def _ConvertHistogramProtoToTuple(histo):
  return {
    "min": histo.min,
    "max": histo.max,
    "num": histo.num,
    "sum": histo.sum,
    "sum_squares": histo.sum_squares,
    "bucket_limit": list(histo.bucket_limit),
    "bucket": list(histo.bucket),
  }

## Normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
## naturally gives bands around median of width 1 std dev, 2 std dev, 3 std dev,
## and then the long tail.
NORMAL_HISTOGRAM_BPS = (0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000)

def _ParseHistogram(tag, wall_time, step, histo):
  histo = _ConvertHistogramProtoToTuple(histo)
  histo_ev = {"wall_time": wall_time, "step": step, "histogram_value": histo}
  return [histo_ev, lambda x: _CompressHistogram(x, NORMAL_HISTOGRAM_BPS)]

def _ParseImage(tag, wall_time, step, image):
  return {
    "wall_time": wall_time,
    "step": step,
    "encoded_image_string": image.encoded_image_string,
    "width": image.width,
    "height": image.height,
  }

def _ParseAudio(tag, wall_time, step, audio):
  return {
    "wall_time": wall_time,
    "step": step,
    "encoded_audio_string": audio.encoded_audio_string,
    "content_type": audio.content_type,
    "sample_rate": audio.sample_rate,
    "length_frames": audio.length_frames,
  }

def _ParseScalar(tag, wall_time, step, scalar):
  return {
    "wall_time": wall_time,
    "step": step,
    "value": scalar,
  }

def _ParseTensor(tag, wall_time, step, tensor):
  return {
    "wall_time": wall_time,
    "step": step,
    "tensor_proto": tensor,
  }

## Different types of summary events handled by the event_accumulator
_SUMMARY_TYPES = {
  'simple_value': _ParseScalar,
  'histo': _ParseHistogram,
  'image': _ParseImage,
  'audio': _ParseAudio,
  'tensor': _ParseTensor,
}

def parse(bytes, wall_time, step, visit):
  summary = summary_pb2.Summary()
  summary.ParseFromString(bytes)

  for value in summary.value:
    if value.HasField('tensor') and value.tag == HEALTH_PILL_EVENT_TAG:
      continue

    for summary_type, summary_func in _SUMMARY_TYPES.items():
      if value.HasField(summary_type):
        datum = getattr(value, summary_type)
        tag = value.node_name if summary_type == 'tensor' else value.tag
        parsed = summary_func(tag, wall_time, step, datum)
        visit(summary_type, parsed)
