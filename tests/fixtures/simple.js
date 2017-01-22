/* @flow */
'use strict';

const fs = require('fs');

const toExport: any = [
  {
    name: "basic graph",
    source: `// comment before
graph main /* comment within */ {
  // comment within
  <- one = 1.0
  /* another one */
}

// comment after`,
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
];

// TODO(adamb) Should error on unused expression
// TODO(adamb) Should error if any function has invalid contents (not only used ones).
const simpleSource = fs.readFileSync(`${__dirname}/simple.nao`).toString()
simpleSource.split("// split").forEach(function(chunk, ix) {
  toExport.push(
    {
      name: `simple construction ${ix.toString()}`,
      action: "test",
      source: chunk.trim(),
    }
  );
})


module.exports = toExport;
