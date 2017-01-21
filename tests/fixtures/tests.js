/* @flow */
'use strict';

module.exports = [
  {
    action: "test",
    name: "simplest fail",
    fails: true,
    source: `graph testAssert {
  tf.Assert(false, {"Assertion failed!"})

  after __leaves { ← result = 0 }
}
`,
    match: /Assertion failed!/,
  },
  {
    action: "test",
    name: "simplest pass",
    source: `graph testAssert {
  tf.Assert(true, {"Assertion failed!"})

  after __leaves { ← result = 0 }
}
`,
    antimatch: /Assertion failed!/,
  },
];
