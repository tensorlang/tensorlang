/* @flow */
'use strict';

var test = require('tape');
var compileString = require('../src/compile').compileString;
var runString = require('../src/run').fromString;
var testString = require('../src/test').fromString;

function compileAndRun(source: string): Promise<string> {
  return new Promise(
    (resolve, reject) => {
      compileString("main", source, null)
      .then((compiled) => {
        runString(compiled, false)
        .then(resolve, reject)
      }, reject);
    }
  );
}

function compileAndTest(source: string): Promise<string> {
  return new Promise(
    (resolve, reject) => {
      compileString("main", source, null)
      .then(
          (compiled) => {
            testString(compiled, false)
            .then(resolve, reject)
          }, reject)
    }
  );
}

var testCases = [
  ...require('./fixtures/simple'),
  ...require('./fixtures/errors'),
  ...require('./fixtures/arguments'),
  ...require('./fixtures/attributes'),
  ...require('./fixtures/tests'),
]

testCases.forEach(
  (tc) => {
    test(tc.name, function (t) {

      var action = tc.action || "run";
      var shouldFail = tc.fails || false;
      var promise;
      switch (action) {
      case "test":
        promise = compileAndTest(tc.source);
        break;
      case "run":
        promise = compileAndRun(tc.source);
        break;
      default:
        t.error(`Unknown action: ${action}`);
        t.end();
        return;
      }

      t.comment(tc.source);

      function checkText(text) {
        if (tc.match) {
          if (!text.match(tc.match)) {
            t.fail(`Output should match ${tc.match.toString()}: ${text}`);
          }
        }

        if (tc.antimatch) {
          if (text.match(tc.antimatch)) {
            t.fail(`Output shouldn't match ${tc.antimatch.toString()}: ${text}`);
          }
        }

        if (tc.expect) {
          t.isEqual(text, tc.expect);
        }
      }

      promise
      .then(
        (str) => {
          checkText(str);
          t.assert(!shouldFail, `Test should fail`);
          t.end();
        },
        (err) => {
          checkText(err.message);

          if (!shouldFail) {
            t.error(err, `Test shouldn't fail`);
          }
          t.end();
        }
      )
    });
  }
)
