/* @flow */
'use strict';

var Server = require('./tensorboard_server');
var electron = require('electron');

var app = electron.app;  // Module to control application life.
var BrowserWindow = electron.BrowserWindow;  // Module to create native browser window.

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
var mainWindow = null;

app.on('ready', function() {
  var devServer = new Server();

  const webPreferences = {
    // Required for some reason, otherwise we get weird errors about d3 not being defined.
    nodeIntegration: false,
  };

  mainWindow = new BrowserWindow({width: 1400, height: 900, webPreferences: webPreferences});
  devServer.indexURL.then((url) => {
    mainWindow.loadURL(url);
  });

  // mainWindow.openDevTools();

  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
    devServer.close();
  });

  mainWindow.webContents.on('did-finish-load', function() {
    mainWindow.setTitle('TensorBoard');
  });
});
