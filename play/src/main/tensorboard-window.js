/* @flow */
'use strict';

import * as os from 'os';
import { BrowserWindow } from 'electron';
import { Server } from './tensorboard-server';

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.

const userDir = os.homedir();

export function openTensorBoard(path) {
  if (!path) {
    path = process.env["NAOPATH"] || process.cwd();
  }

  var server = new Server([
    "--reopen-stderr", "/Users/adamb/debug.log",
    "--reopen-stdout", "/Users/adamb/debug.log",
    "--workspace", path,
  ]);

  var prettyPath = path.replace(userDir, "~");
  const windowTitle = `${prettyPath} — TensorBoard`;

  const mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    title: windowTitle,
    webPreferences: {
      // Required to avoid weird errors about d3 not being defined.
      nodeIntegration: false,
    }
  });
  server.start().then((url) => {
    mainWindow.loadURL(url);
  });

  mainWindow.setRepresentedFilename(path);

  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    server.close();
  });

  mainWindow.on('page-title-updated', function(event) {
    event.preventDefault();
  });

  return mainWindow;
}
