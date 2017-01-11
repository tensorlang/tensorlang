/* @flow */
"use strict";

const spawn = require('child_process').spawn;
const stream = require('stream');

var exitCodeToError = function(code) {
  return code === 0 ? null : new Error("Exit code was " + code);
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
  withStdinCapturingStdout: function(cmd: string, args: string[], stdinString: string, callback: (error: ?Error, str: string) => void) {
    var data = []; // We'll store all the data inside this array
    var converter = new stream.Writable({
      write(chunk: Buffer, encoding: string, callback: (err: ?Error) => any) {
        data.push(chunk);
      }
    });

    var process = spawn(cmd, args, {env: subprocessEnv, stdio: ['pipe', 'pipe', 2]});
    process.stdout.pipe(converter);

    process.stdin.write(stdinString);
    process.stdin.end();

    process.on('exit', function(code, signal) {
      // Create a buffer from all the received chunks
      var b = Buffer.concat(data);
      callback(exitCodeToError(code), b.toString());
    });
  }
};
