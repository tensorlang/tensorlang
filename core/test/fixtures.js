/* @flow */
'use strict';

const test = require('tape');
const tmp = require('tmp');
const fs = require('fs');
const path = require('path');
const spawnProcess = require('./util/spawnProcess');

const testCases = [
  ...require('./fixtures/imports'),
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

      var workspaceTmpDir;
      if (tc.sources) {
        // Create temporary directory.
        workspaceTmpDir = tmp.dirSync({unsafeCleanup: true});
        additionalArgs.push("--workspace", workspaceTmpDir.name);

        const srcDir = path.join(workspaceTmpDir.name, "src");
        fs.mkdirSync(srcDir)
        // Populate directory.
        for (const key in tc.sources) {
          const content = tc.sources[key];
          const p = path.join(srcDir, key);
          fs.writeFileSync(p, content);
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
          workspaceTmpDir && workspaceTmpDir.removeCallback();
        },
        (err) => {
          checkText(err.message);

          if (!shouldFail) {
            t.error(err, `Test shouldn't fail`);
          }
          t.end();
          workspaceTmpDir && workspaceTmpDir.removeCallback();
        }
      )
    });
  }
)
