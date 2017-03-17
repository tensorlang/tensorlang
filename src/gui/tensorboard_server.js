/* @flow */
"use strict";

const spawn = require('child_process').spawn;
const process = require('process');

const net = require('net');

// attemptConnect returns a Promise that resolves to true if a TCP connection
// could be made, false otherwise.
function attemptConnect(host, port) {
  return new Promise((resolve) => {
    var client = new net.Socket();
    client.connect(port, host, function() {
      resolve(true);
      client.destroy();
    });

    client.on('error', function() {
      resolve(false);
    })
  });
}

// awaitReady returns a Promise that resolves when a TCP connection can be
// made to the given `host` and `port`. If a connection fails it is retried
// after `retryDelay` milliseconds.
function awaitReady(host, port, retryDelay) {
  return new Promise((resolve, reject) => {
    function attempt() {
      attemptConnect(host, port).then((success) => {
        if (success) {
          resolve();
        } else {
          setTimeout(attempt, retryDelay);
        }
      });
    };
    attempt();
  });
}

class Server {
  constructor(baseArgs) {
    const host = "127.0.0.1";
    const port = 6006;
    const args = [];
    if (baseArgs) {
      args.concat(baseArgs);
    }

    args.push("--reopen-stderr", "/Users/adamb/debug.log");
    args.push("--tensorboard", `${host}:${port}`);

    const env = {
      "PYTHONUNBUFFERED": "1",
      "NAOPATH": process.env["NAOPATH"]
    };

    this._process = spawn(`${__dirname}/../../bin/nao`, args, {env: env});
    this._process.stdout.pipe(process.stdout);
    this._process.stderr.pipe(process.stderr);

    this.indexURL = awaitReady(host, port, 250).then(() => `http://${host}:${port}`)
  }

  close() {
    this._process.kill();
  }
}

Server.forLogDir = function(logDir) {
  return new Server(["--log-dir", logDir]);
}

Server.forWorkspace = function(workspaceDir) {
  return new Server(["--workspace", workspaceDir]);
}

module.exports = Server;
