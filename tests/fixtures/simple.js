/* @flow */
'use strict';

const fs = require('fs');

module.exports = [
  {
    name: "basic graph",
    source: `graph main { <- one scalar float = 1.0 }`,
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
  producer: 17
}
`
  },
  // TODO(adamb) Should error on unused expression
  // TODO(adamb) Should error if any function has invalid contents (not only used ones).
  {
    name: "simple constructions",
    action: "test",
    source: fs.readFileSync(`${__dirname}/simple.nao`).toString(),
  }
];
