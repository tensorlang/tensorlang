/* @flow */
'use strict';

const TensorBoard = require('./tensorboard');

var electron = require('electron');

var app = electron.app;  // Module to control application life.

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
var mainWindow = null;

app.on('ready', function() {
  var path = process.env["NAOPATH"];
  if (!path) {
    path = process.cwd();
  }

  mainWindow = TensorBoard.openTensorBoard(path);
  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
  });
});
