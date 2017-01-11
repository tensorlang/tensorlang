/* @flow */
"use strict";

const parseExpressions = require('./compile/parseExpressions.js');
const spawnProcess = require('./util/spawnProcess.js');

module.exports = {
  fromString: function(graph: string):Promise<string> {
    return new Promise((resolve, reject) => {
      spawnProcess.withStdinCapturingStdout(
        "../bin/python",
        [
          `${__dirname}/cli.py`,
          "--graphdef", "/dev/stdin",
          "--run"
        ],
        graph,
        (err, str) => { err ? reject(err) : resolve(str); }
      );
    });
  },
  fromFile: function(file: string, binary: boolean):Promise<void> {
    return new Promise((resolve, reject) => {
      spawnProcess.withStdin(
        "../bin/python",
        [
          `${__dirname}/cli.py`,
          "--graphdef", file,
          ...(binary ? ["--binaryGraphdef"] : []),
          "--run"
        ],
        "",
        (err) => { err ? reject(err) : resolve(); }
      );
    });
  }
};
