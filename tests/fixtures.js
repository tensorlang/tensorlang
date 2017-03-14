/* @flow */
'use strict';

var test = require('tape');
var spawnProcess = require('./util/spawnProcess');

var testCases = [
  ...require('./fixtures/simple'),
  ...require('./fixtures/loop'),
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
      var additionalArgs = [];
      var promise;
      switch (action) {
      case "test":
        additionalArgs.push("--test");
        break;
      case "run":
        additionalArgs.push("--run");
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

      spawnProcess.withStdinCapturingStdout(
        `${__dirname}/../bin/nao`,
        [
          "--source", tc.source,
          ...additionalArgs,
        ],
        ""
      )
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
