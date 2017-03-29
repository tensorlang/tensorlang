/* @flow */
'use strict';

module.exports = [
//   {
//     name: "require keyword argument ellipsis",
//     fails: true,
//     source: `func lotsOfAttributes[a, b]() {
//       <- result = a + b
//     }
//
//     func TestAttributes() {
//       lotsOfAttributes[a: 1.0] -- foo
//       foo[b: 2.0]  -- bar
//       <- result = bar()
//     }
// `,
//     action: "test",
//     match: /missing attributes/
//   },
//   {
//     name: "forbid argument ellipsis in apply",
//     fails: true,
//     source: `func lotsOfAttributes[a, b]() {
//       <- result = a + b
//     }
//
//     func main() {
//       <- result = lotsOfAttributes[a: 1.0, b: 2.0, ...]()
//     }
// `,
//   },
//   {
//     name: "can't redefine keyword arguments",
//     action: "test",
//     fails: true,
//     match: /already defined/,
//     source: `func lotsOfAttributes[a, b, c]() {
//       <- result = a + b + c
//     }
//
//     func TestAttributes() {
//       lotsOfAttributes[a: 1.0, b: 2.0, ...] -- foo
//       foo[b: 3.0, c: 4.0] -- bar
//
//       after __leaves { ← result = 0 }
//     }
// `,
//   },
// {
//     name: "use keyword arg ellipsis",
//     source: `func lotsOfAttributes[a, b, c]() {
//       <- result = a + b + c
//     }
//
//     func TestAttributes() {
//       lotsOfAttributes[a: 1.0, b: 2.0, ...] -- foo
//       foo[c: 3.0]  -- bar
//
//       tf.Assert(6.0 == bar(), {"Assertion failed!"})
//
//       after __leaves { ← result = 0 }
//     }
// `,
//     action: "test",
//   },
  {
    name: "simple keyword arguments",
    source: `func lotsOfArgs(a, b, c) {
      <- result = a / b - c
    }

    func TestAttributes() {
      lotsOfArgs(a: 4.0, c: 1.0, b: 2.0)
      tf.Assert(1.0 == ^, {"Assertion failed!"})

      after __leaves { ← result = 0 }
    }
`,
    action: "test",
  },
  {
    name: "no repeats allowed for keyword arguments",
    fails: true,
    source: `func lotsOfArgs(a, b, c) {
      <- result = a / b - c
    }

    func TestAttributes() {
      lotsOfArgs(a: 4.0, c: 1.0, b: 2.0, c: 1.0)
      tf.Assert(1.0 == ^, {"Assertion failed!"})

      after __leaves { ← result = 0 }
    }
`,
    action: "test",
  },
];
