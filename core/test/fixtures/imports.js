/* @flow */
'use strict';

module.exports = [
  {
    name: "simplest nao import",
    source: `import (
  "one"
)

func Main() {
  emit a = one.One
  emit b = one.One
}
`,
    sources: {
      "one.nao": `
let One = 0.0
`,
    },
    expect: `node {
  name: "one"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
versions {
  producer: 21
}
`
  },
  {
    name: "simple python and nao imports",
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
  {
    name: "different nao imports can reference the same variable",
    action: "test",
    source: `import (
  "some_var"
  "some_var_addition"
  "some_var_multiplication"
)

func TestImports() {
  tf.Assert(some_var.Get() == 0.0, {"some_python.Identity(123.0) == 123.0"})
  after __leaves {
    tf.Assert(some_var.Set(1.0) == 1.0, {"some_python.Identity(123.0) == 123.0"})
    after __leaves {
    tf.Assert(some_var.Get() == 1.0, {"some_python.Identity(123.0) == 123.0"})

  }
  tf.Assert(some_nao.Identity(123.0) == 123.0, {"some_nao.Identity(123.0) == 123.0"})
  tf.Assert(some_nao.Increment(122.0) == 123.0, {"some_nao.Increment(122.0) == 123.0"})
  tf.Assert(some_nao.True, {"some_nao.True"})
  tf.Assert(some_nao.Zero == 0.0, {"some_nao.Zero == 0.0"})

  <- x = after __leaves { 0 }
}
`,
    sources: {
      "some_var.nao": `
var v = 0
func Get() {
  emit v = v
}

func Set(val) {
  v = val
  emit r = v
}`,
      "some_var_addition.nao": `
import "some_var"

func IncrementBy(i) {
  emit r = some_var.Set(some_var.Get() + i)
}
`,
      "some_var_multiplication.nao": `
import "some_var"

func ScaleBy(i) {
  emit r = some_var.Set(some_var.Get() * i)
}
`,
    }
  }
//   {
//     name: "test import and non-exported values. update the below to fail during compilation if any function with an initial char that's lowercase is imported.",
//     source: `
// import some_python
// import some_nao
//
// func TestImports() {
//   tf.Assert(some_python.Identity(123.0) == 123.0, {"some_python.Identity(123.0) == 123.0"})
//   tf.Assert(some_nao.Identity(123.0) == 123.0, {"some_nao.Identity(123.0) == 123.0"})
//   tf.Assert(some_nao.Zero == 0.0, {"some_nao.Zero == 0.0"})
// }
// `,
//     sources: {
//       "some_python.py": `
// from tensorflow.python.framework import dtypes
//
// def identity(n) -> dtypes.float32:
//   return n
// `,
//       "some_nao.nao": `
// let zero = 0.0
// func identity(n) { emit x = n }
// `,
//     }
//   },


//   {
//     name: "import and await training",
//     source: `
// import some_python
// import some_nao
//
// func TestImports() {
//   tf.Assert(some_python.Identity(123.0) == 123.0, {"some_python.Identity(123.0) == 123.0"})
//   tf.Assert(some_nao.Identity(123.0) == 123.0, {"some_nao.Identity(123.0) == 123.0"})
//   tf.Assert(some_nao.Zero == 0.0, {"some_nao.Zero == 0.0"})
// }
// `,
//     sources: {
//       "some_python.py": `
// from tensorflow.python.framework import dtypes
//
// def Identity(n) -> dtypes.float32:
//   return n
// `,
//       "some_nao.nao": `
// let Zero = 0.0
// func Identity(n) { emit x = n }
// `,
//     }
//   },
];
