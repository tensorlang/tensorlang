/* @flow */
'use strict';

module.exports = [
  {
    name: "empty source",
    source: `
`,
    fails: true,
    match: /end of input/,
  },
];
