/* @flow */
'use strict';

module.exports = [
  {
    name: "loop ???",
    action: "test",
    source: `import nn "tensorflow:nn"

tref conv1_weights float <5, 5, 1, 32> = tf.truncated_normal[shape: <5, 5, 1, 32>, stddev: 0.1]()
func sum(b, w) {
  b
  nn.conv2d[
    strides: [1, 1, 1, 1],
    padding: "SAME",
  ](^, w)
  emit s = ^
}

graph testLoop {
  out = rec x = 1; x <= 5 {
    sum(tf.truncated_normal[shape: <1, 28, 28, 1>, stddev: 0.1](), conv1_weights)

    <- x = after __leaves { x + 1 }
  }

  tf.Assert(out:x == 6, {"out:x == 6"})

  ← result = after __leaves { 0 }
}
`,
  },
  {
    name: "loop ???",
    action: "test",
    source: `import nn "tensorflow:nn"

tref conv1_weights float <5, 5, 1, 32> = tf.truncated_normal[shape: <5, 5, 1, 32>, stddev: 0.1]()
func sum(b) {
  b
  nn.conv2d[
    strides: [1, 1, 1, 1],
    padding: "SAME",
  ](^, conv1_weights)
  emit s = ^
}

graph testLoop {
  out = rec x = 1; x <= 5 {
    sum(tf.truncated_normal[shape: <1, 28, 28, 1>, stddev: 0.1]())

    <- x = after __leaves { x + 1}
  }

  tf.Assert(out:x == 6, {"out:x == 6"})

  ← result = after __leaves { 0 }
}
`,
  },
  {
    name: "loop ???",
    action: "test",
    source: `
func sum(a, b) {
  emit s = a + b
}

graph testLoop {
  tref a int32<> = 1
  out = rec x = 1; x <= 5 {
    a + 5

    <- x = sum(x, 1)
  }

  tf.Assert(out:x == 6, {"out:x == 6"})

  ← result = after __leaves { 0 }
}
`,
  },
  {
    name: "simple loop",
    action: "test",
    source: `
graph testLoop {
  out = rec x = 1; x <= 5 {
    <- x = x + 1
  }

  tf.Assert(out:x == 6, {"out:x == 6"})

  ← result = after __leaves { 0 }
}
`,
  },
  {
    name: "loop calling a function in cond",
    action: "test",
    source: `
func isFiveOrLess(a) {
  emit s = a <= 5
}

graph testLoop {
  out = rec x = 1; isFiveOrLess(x) {
    <- x = x + 1
  }

  tf.Assert(out:x == 6, {"out:x == 6"})

  ← result = after __leaves { 0 }
}
`,
  },
  {
    name: "loop calling a function in body",
    action: "test",
    source: `
func sum(a, b) {
  emit s = a + b
}

graph testLoop {
  out = rec x = 1; x <= 5 {
    <- x = sum(x, 1)
  }

  tf.Assert(out:x == 6, {"out:x == 6"})

  ← result = after __leaves { 0 }
}
`,
  },
  {
    name: "loop using a variable from outside the loop as an arg to a function",
    action: "test",
    source: `
func sum(a, b) {
  emit s = a + b
}

graph testLoop {
  tref a int32<> = 1
  out = rec x = 1; x <= 5 {
    <- x = sum(x, a)
  }

  tf.Assert(out:x == 6, {"out:x == 6"})

  ← result = after __leaves { 0 }
}
`,
  },
  {
    name: "loop using a variable inside the loop as an arg to a function",
    action: "test",
    source: `
graph testLoop {
  out = rec x = 1.0; y float <700> = 0; x <= 5.0 {
    tref q float<700> = tf.truncated_normal[shape: <700>]()
    func () {
      emit s = q + 1.0
    } -- unity

    <- y = unity()
    <- x = x + 1.0
  }

  tf.Assert(out:x == 6.0, {"out:x == 6.0"})

  ← result = after __leaves { 0 }
}
`,
  },
];
