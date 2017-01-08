const parseExpressions = require('./compile/parse-expressions');

const fs = require('fs');
const spawn = require('child_process').spawn;
const stream = require('stream');

const env = process.env;
const CUDA_HOME = "/usr/local/cuda";
const subprocessEnv = {
  "CUDA_HOME": CUDA_HOME,
  "DYLD_LIBRARY_PATH": `${env.DYLD_LIBRARY_PATH}:${CUDA_HOME}/lib`,
  "PATH": `${CUDA_HOME}/bin:${env.PATH}`,
};

var exitCodeToError = function(code) {
  return code === 0 ? null : new Error("Exit code was " + code);
};

var spawnProcessWithStdin = function(env, cmd, args, stdinString, callback) {
  var process = spawn(cmd, args, {env: env, stdio: ['pipe', 1, 2]});
  process.stdin.write(stdinString);
  process.stdin.end();
  process.on('end', function(code, signal) {
    callback(exitCodeToError(code));
  });
};

var spawnProcessWithStdinCapturingStdout = function(env, cmd, args, stdinString, callback) {
  var converter = new stream.Writable();
  converter.data = []; // We'll store all the data inside this array
  converter._write = function (chunk) {
    this.data.push(chunk);
  };

  var process = spawn(cmd, args, {env: env, stdio: ['pipe', 'pipe', 2]});
  process.stdout.pipe(converter);

  process.stdin.write(stdinString);
  process.stdin.end();

  process.on('end', function(code, signal) {
    // Create a buffer from all the received chunks
    var b = Buffer.concat(converter.data);
    callback(exitCodeToError(code), b.toString());
  });
};

var compileString = function(source, callback) {
  spawnProcessWithStdinCapturingStdout(
    subprocessEnv,
    "../bin/python",
    [
      `${__dirname}/json2pb.py`,
      "--input_json", "/dev/stdin",
      "--output_graph", "/dev/stdout",
    ],
    JSON.stringify(parseExpressions(source)),
    callback
  );
};

var compile = function(input, output, outputBinary, callback) {
  // TODO(adamb) Don't do this synchronously
  var source = fs.readFileSync(input).toString();

  spawnProcessWithStdin(
    subprocessEnv,
    "../bin/python",
    [
      `${__dirname}/json2pb.py`,
      "--input_json", "/dev/stdin",
      "--output_graph", output,
      ...(outputBinary ? ["--output_binary"] : [])
    ],
    JSON.stringify(parseExpressions(source)),
    callback
  );
};

module.exports = {
  compile: compile,
  compileString: compileString,
};
