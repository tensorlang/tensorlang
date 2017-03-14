/* @flow */
'use strict';

module.exports = [
  {
    name: "require attribute ellipsis",
    fails: true,
    source: `func lotsOfAttributes[a, b]() {
      <- result = a + b
    }

    func TestAttributes() {
      lotsOfAttributes[a: 1.0] -- foo
      foo[b: 2.0]  -- bar
      <- result = bar()
    }
`,
    action: "test",
    match: /missing attributes/
  },
  {
    name: "forbid attribute ellipsis in apply",
    fails: true,
    source: `func lotsOfAttributes[a, b]() {
      <- result = a + b
    }

    func Main() {
      <- result = lotsOfAttributes[a: 1.0, b: 2.0, ...]()
    }
`,
  },
  {
    name: "can't redefine attributes",
    action: "test",
    fails: true,
    match: /already defined/,
    source: `func lotsOfAttributes[a, b, c]() {
      <- result = a + b + c
    }

    func TestAttributes() {
      lotsOfAttributes[a: 1.0, b: 2.0, ...] -- foo
      foo[b: 3.0, c: 4.0] -- bar

      after __leaves { ← result = 0 }
    }
`,
  },
  {
    name: "use attribute ellipsis",
    source: `func lotsOfAttributes[a, b, c]() {
      <- result = a + b + c
    }

    func TestAttributes() {
      lotsOfAttributes[a: 1.0, b: 2.0, ...] -- foo
      foo[c: 3.0]  -- bar

      tf.Assert(6.0 == bar(), {"Assertion failed!"})

      after __leaves { ← result = 0 }
    }
`,
    action: "test",
  },
  {
    name: "can't use ellipsis in function apply",
    fails: true,
    source: `func lotsOfAttributes[a, b, c]() {
      <- result = a + b + c
    }

    func TestAttributes() {
      lotsOfAttributes[a: 1.0, b: 2.0, ...] -- foo
      tf.Assert(6.0 == foo[c: 3.0, ...](), {"Assertion failed!"})

      after __leaves { ← result = 0 }
    }
`,
    action: "test",
  },
  {
    name: "simple attribute use",
    fails: true,
    source: `func lotsOfAttributes[a, b, c]() {
      <- result = c / b - a
    }

    func TestAttributes() {
      lotsOfAttributes[a: 1.0, b: 2.0, c: 4.0] -- foo
      tf.Assert(1.0 == ^](), {"Assertion failed!"})

      after __leaves { ← result = 0 }
    }
`,
    action: "test",
  },
  {
    name: "no repeats in attributes",
    fails: true,
    source: `func lotsOfAttributes[a, b, c]() {
      <- result = c / b - a
    }

    func TestAttributes() {
      lotsOfAttributes[a: 1.0, b: 2.0, c: 4.0, b: 2.0] -- foo
      tf.Assert(1.0 == ^](), {"Assertion failed!"})

      after __leaves { ← result = 0 }
    }
`,
    action: "test",
  },
];
