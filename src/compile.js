/* @flow */
"use strict";

const parseExpressions = require('./compile/parseExpressions.js');
const spawnProcess = require('./util/spawnProcess.js');

module.exports = {
  compileString: function(source: string): Promise<string> {
    var expressions;
    try {
      expressions = parseExpressions(source);
    } catch (e) {
      return Promise.reject(e);
    }

    return spawnProcess.withStdinCapturingStdout(
      "../bin/python",
      [
        `${__dirname}/cli.py`,
        "--input-json", "/dev/stdin",
        "--output-metagraphdef", "/dev/stdout",
      ],
      JSON.stringify(expressions)
    );
  },
  compile: function(source: string, output: string, binary: boolean): Promise<any> {
    var expressions;
    try {
      expressions = parseExpressions(source);
    } catch (e) {
      return Promise.reject(e);
    }

    return new Promise((resolve, reject) => {
      spawnProcess.withStdin(
        "../bin/python",
        [
          `${__dirname}/cli.py`,
          "--input-json", "/dev/stdin",
          ...(output ? ["--output-metagraphdef", output] : []),
          ...(binary ? ["--output-binary"] : [])
        ],
        JSON.stringify(expressions),
        (err) => { err ? reject(err) : resolve(); }
      );
    });
  }
};
