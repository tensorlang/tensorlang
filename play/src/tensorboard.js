/* @flow */
'use strict';

const os = require('os');
const electron = require('electron');

const BrowserWindow = electron.BrowserWindow;  // Module to create native browser window.

const Server = require('./tensorboard_server');

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.

const userDir = os.homedir();

function openTensorBoard(path) {
  var path = process.env["NAOPATH"];
  if (!path) {
    path = process.cwd();
  }

  var devServer = new Server([
    "--reopen-stderr", "/Users/adamb/debug.log",
    "--reopen-stdout", "/Users/adamb/debug.log",
    "--workspace", path,
  ]);

  const webPreferences = {
    // Required for some reason, otherwise we get weird errors about d3 not being defined.
    nodeIntegration: false,
  };

  var p = path.replace(userDir, "~");
  const windowTitle = `${p} — TensorBoard`;

  const mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    title: windowTitle,
    webPreferences: webPreferences
  });
  devServer.start().then((url) => {
    mainWindow.loadURL(url);
  });


  mainWindow.setRepresentedFilename(path);
  mainWindow.setTitle(windowTitle);

  // mainWindow.openDevTools();

  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    devServer.close();
  });

  mainWindow.on('page-title-updated', function(event) {
    event.preventDefault();
  });

  return mainWindow;
}

exports.openTensorBoard = openTensorBoard;
