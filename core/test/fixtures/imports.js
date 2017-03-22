/* @flow */
'use strict';

module.exports = [
  {
    name: "import",
    action: "test",
    source: `import (
  "some_python"
  "some_nao"
)

func TestImports() {
  tf.Assert(some_python.Identity(123.0) == 123.0, {"some_python.Identity(123.0) == 123.0"})
  tf.Assert(some_nao.Identity(123.0) == 123.0, {"some_nao.Identity(123.0) == 123.0"})
  tf.Assert(some_nao.Increment(122.0) == 123.0, {"some_nao.Increment(122.0) == 123.0"})
  tf.Assert(some_nao.True, {"some_nao.True"})
  tf.Assert(some_nao.Zero == 0.0, {"some_nao.Zero == 0.0"})

  <- x = after __leaves { 0 }
}
`,
    sources: {
      "some_python.py": `
from tensorflow.python.framework import dtypes

def Identity(n) -> dtypes.float32:
  return n
`,
      "some_nao.nao": `
let True = true
let Zero = 0.0
func Identity(n float) { emit x = n }
func Increment(n float) { emit x = n + 1.0 }
`,
    }
  },
];
