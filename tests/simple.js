'use strict';

// Test parsing simple expression
// Test generating simple IR
// Use python program to convert simple IR to .pbtxt
// Use different python program to use .pbtxt to run graph with given data

var test = require('tape-catch');
var compile = require('../src/compile').compileString;
const fs = require('fs');

test('simple test', function (t) {
  var src = fs.readFileSync(`${__dirname}/../examples/identity.nao`);
  compile(src.toString(), function(err, result) {
    if (err) {
      t.end(err);
      return;
    }

    console.log(result);
  });
});
