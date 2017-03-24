/* @flow */
import path from 'path';

import { shell, BrowserWindow, ipcMain as ipc } from 'electron';

let launchIpynb;

export function getPath(url) {
  const nUrl = url.substring(url.indexOf('static'), path.length);
  return path.join(__dirname, '..', '..', nUrl.replace('static/', ''));
}

export function deferURL(event, url) {
  event.preventDefault();
  if (!url.startsWith('file:')) {
    shell.openExternal(url);
  } else if (url.endsWith('.ipynb')) {
    launchIpynb(getPath(url));
  }
}

const iconPath = path.join(__dirname, '..', '..', 'static', 'icon.png');

const initContextMenu = require('electron-context-menu');

// Setup right-click context menu for all BrowserWindows
initContextMenu();

export function launchNotebookFromFile(filename) {
  let win = new BrowserWindow({
    width: 800,
    height: 1000,
    icon: iconPath,
    title: 'Untitled - loading',
  });

  const index = path.join(__dirname, '..', '..', 'static', 'notebook.html');
  win.loadURL(`file://${index}`);

  win.webContents.on('did-finish-load', () => {
    if (filename) {
      win.webContents.send('main:load', filename);
    }
    win.webContents.send('main:load-config');
  });

  win.webContents.on('will-navigate', deferURL);

  // Emitted when the window is closed.
  win.on('closed', () => {
    win = null;
  });
  return win;
}
launchIpynb = launchNotebookFromFile;

const defaultKernelSpec = {
  name: 'nao',
  spec: {
    argv: [
      "/Users/adamb/github/ajbouh/nao/core/build/exe.macosx-10.6-x86_64-3.5/bin/nao",
      "--reopen-stderr", "/Users/adamb/debug.log",
      "--reopen-stdout", "/Users/adamb/debug.log",
      "--jupyter-kernel",
      '{connection_file}'
    ],
    display_name: 'Nao',
    language: 'nao'
  }
};

const defaultKernelSpecs = {}
defaultKernelSpecs[defaultKernelSpec.name] = defaultKernelSpec;

ipc.on('kernel_specs_request', (event) => {
  event.sender.send('kernel_specs_reply', defaultKernelSpecs);
});

export function launchNewNotebook(kernelSpec) {
  kernelSpec = kernelSpec || defaultKernelSpec;
  const win = launchNotebookFromFile();
  win.webContents.on('did-finish-load', () => {
    win.webContents.send('main:new', kernelSpec);
    win.send('menu:set-blink-rate', 530);
  });
  return win;
}
