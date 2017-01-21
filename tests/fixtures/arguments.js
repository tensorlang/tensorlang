/* @flow */
'use strict';

module.exports = [
  {
    name: "simple keyword arguments",
    source: `func lotsOfArgs(a, b, c) {
      <- result = a / b - c
    }

    graph testAttributes {
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

    graph testAttributes {
      lotsOfArgs(a: 4.0, c: 1.0, b: 2.0, c: 1.0)
      tf.Assert(1.0 == ^, {"Assertion failed!"})

      after __leaves { ← result = 0 }
    }
`,
    action: "test",
  },
];
