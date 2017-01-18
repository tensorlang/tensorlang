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

const env = process.env;
const CUDA_HOME = "/usr/local/cuda";
const subprocessEnv = {
  "CUDA_HOME": CUDA_HOME,
  "DYLD_LIBRARY_PATH": `${env['DYLD_LIBRARY_PATH'] || ''}:${CUDA_HOME}/lib`,
  "PATH": `${CUDA_HOME}/bin:${env['PATH'] || ''}`,
};

module.exports = {
  withStdin: function(cmd: string, args: string[], stdinString: string, callback: (error: ?Error) => void) {
    var process = spawn(cmd, args, {env: subprocessEnv, stdio: ['pipe', 1, 2]});
    process.stdin.write(stdinString);
    process.stdin.end();
    process.on('exit', function(code, signal) {
      callback(exitCodeToError(code));
    });
  },
  withStdinCapturingStdout: function(cmd: string, args: string[], stdinString: string): Promise<string> {
    return new Promise((resolve, reject) => {
      var process = spawn(cmd, args, {env: subprocessEnv, stdio: ['pipe', 'pipe', 'pipe']});

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
  withStdinCapturingStdoutAndStderr: function(cmd: string, args: string[], stdinString: string): Promise<string> {
    return new Promise((resolve, reject) => {
      var process = spawn(cmd, args, {env: subprocessEnv, stdio: ['pipe', 'pipe', 'pipe']});

      var data = []; // We'll store all the data inside this array
      process.stdout.on('data', function(chunk) { data.push(chunk); });
      process.stderr.on('data', function(chunk) { data.push(chunk); });

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
          var b = Buffer.concat(data);
          var output = b.toString();

          var err = exitCodeToError(exitCode, output);

          if (err) {
            reject(err);
          } else {
            resolve(output);
          }
        }
      );

      process.stdin.write(stdinString);
      process.stdin.end();
    });
  },
};
