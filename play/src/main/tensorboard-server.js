/* @flow */
"use strict";

import { spawn } from 'child_process';
import { env, stdout, stderr } from 'process';
import { Socket, createServer } from 'net';

// attemptConnect returns a Promise that resolves to true if a TCP connection
// could be made, false otherwise.
function attemptConnect(host, port) {
  return new Promise((resolve) => {
    var client = new Socket();
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

// selectPort returns a promise that resolves to a recently free port.
function selectPort() {
  return new Promise((resolve, reject) => {
    var server = createServer();
    server.on('listening', function() {
      const port = server.address().port;
      console.log('listening', port);
      server.close();
      resolve(port);
    });
    server.on('error', function(err) {
      reject(err);
    });
    server.listen(0);
  });
}

export class Server {
  constructor(baseArgs) {
    this._baseArgs = baseArgs;
  }

  start() {
    const args = [];
    if (this._baseArgs) {
      args.push(...this._baseArgs);
    }

    var nao = env["NAO"];
    if (!nao) {
      nao = `${__dirname}/../../../core/build/exe.macosx-10.6-x86_64-3.5/bin/nao`;
    }

    return selectPort()
    .then((port) => {
      const host = "127.0.0.1";
      args.push("--tensorboard", `${host}:${port}`);

      this._process = spawn(
        nao,
        args,
        {
          env: {
            "PYTHONUNBUFFERED": "1",
          }
        }
      );
      this._process.stdout.pipe(stdout);
      this._process.stderr.pipe(stderr);

      return awaitReady(host, port, 250).then(() => `http://${host}:${port}`);
    });
  }

  close() {
    this._process.kill();
  }
}
