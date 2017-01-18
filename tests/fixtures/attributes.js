/* @flow */
'use strict';

module.exports = [
  {
    name: "require attribute ellipsis",
    fails: true,
    source: `func lotsOfAttributes {
      @@ a
      @@ b

      <- result scalar float = a + b
    }

    graph testAttributes {
      lotsOfAttributes[a: 1.0] -- foo
      foo[b: 2.0]  -- bar
      <- result scalar float = bar()
    }
`,
    action: "test",
    match: /missing attributes/
  },
  {
    name: "forbid attribute ellipsis in apply",
    fails: true,
    source: `func lotsOfAttributes {
      @@ a
      @@ b

      <- result scalar float = a + b
    }

    graph main {
      <- result scalar float = lotsOfAttributes[a: 1.0, b: 2.0, ...]()
    }
`,
  },
  {
    name: "can't redefine attributes",
    action: "test",
    fails: true,
    match: /already defined/,
    source: `func lotsOfAttributes {
      @@ a
      @@ b
      @@ c

      <- result scalar float = a + b + c
    }

    graph testAttributes {
      lotsOfAttributes[a: 1.0, b: 2.0, ...] -- foo
      foo[b: 3.0, c: 4.0] -- bar

      after __leaves { ← result scalar int8 = tf.identity(0) }
    }
`,
  },
  {
    name: "use attribute ellipsis",
    source: `func lotsOfAttributes {
      @@ a
      @@ b
      @@ c

      <- result scalar float = a + b + c
    }

    graph testAttributes {
      lotsOfAttributes[a: 1.0, b: 2.0, ...] -- foo
      foo[c: 3.0]  -- bar

      tf.Assert(6.0 == bar(), {"Assertion failed!"})

      after __leaves { ← result scalar int8 = tf.identity(0) }
    }
`,
    action: "test",
  },
];
