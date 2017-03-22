/* @flow */
"use strict";

const spawn = require('child_process').spawn;
const stream = require('stream');

var exitCodeToError = function(code, output) {
  var msg = "Exit code was " + code;
  if (output){
     msg = msg + ". Output " + output;
  }
  return code === 0 ? null : new Error(msg);
};

module.exports = {
  withStdinCapturingStdout: function(cmd: string, args: string[], stdinString: string): Promise<string> {
    return new Promise((resolve, reject) => {
      var process = spawn(cmd, args, {stdio: ['pipe', 'pipe', 'pipe']});

      var stdoutData = [];
      process.stdout.on('data', function(chunk) { stdoutData.push(chunk); });

      var stderrData = [];
      process.stderr.on('data', function(chunk) { stderrData.push(chunk); });

      var exitCode;

      Promise.all(
        [
          new Promise((rs, rj) => { process.stderr.on('end', rs.bind(null, null)); }),
          new Promise((rs, rj) => { process.stdout.on('end', rs.bind(null, null)); }),
          new Promise((rs, rj) => {
            process.on('exit', function(code, signal) { exitCode = code; rs(); });
          })
        ]
      )
      .then(
        () => {
          var err = exitCodeToError(exitCode, Buffer.concat(stderrData).toString());

          if (err) {
            reject(err);
          } else {
            resolve(Buffer.concat(stdoutData).toString());
          }
        }
      );

      process.stdin.write(stdinString);
      process.stdin.end();
    });
  },
};
