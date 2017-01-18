/* @flow */
"use strict";

const spawnProcess = require('./util/spawnProcess.js');

module.exports = {
  fromString: function(graph: string):Promise<string> {
    return spawnProcess.withStdinCapturingStdoutAndStderr(
      "../bin/python",
      [
        `${__dirname}/cli.py`,
        "--graphdef", "/dev/stdin",
        "--test"
      ],
      graph
    );
  },
  fromFile: function(file: string, binary: boolean):Promise<void> {
    return new Promise((resolve, reject) => {
      spawnProcess.withStdin(
        "../bin/python",
        [
          `${__dirname}/cli.py`,
          "--graphdef", file,
          ...(binary ? ["--binary-raphdef"] : []),
          "--test"
        ],
        "",
        (err) => { err ? reject(err) : resolve(); }
      );
    });
  }
};
