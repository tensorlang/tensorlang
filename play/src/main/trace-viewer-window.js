/* @flow */
import path from 'path';

import { shell, BrowserWindow } from 'electron';

const iconPath = path.join(__dirname, '..', '..', 'static', 'icon.png');

const index = path.join(__dirname, '..', '..', 'static', 'trace-viewer.html');
const preload = path.join(__dirname, '..', '..', 'static', 'trace-viewer-preload.js');

export function launchTraceViewerForFile(filename) {
  let win = new BrowserWindow({
    width: 800,
    height: 1000,
    icon: iconPath,
    title: `${filename} — Trace Viewer`,
    webPreferences: {
      // Required to avoid weird errors about d3 not being defined.
      preload: preload,
      nodeIntegration: false,
    }
  });

  win.loadURL(`file://${index}`);

  win.webContents.on('did-finish-load', () => {
    if (filename) {
      win.webContents.send('main:load', filename);
    }
  });

  win.on('page-title-updated', function(event) {
    event.preventDefault();
  });

  // Emitted when the window is closed.
  win.on('closed', () => {
    win = null;
  });

  return win;
}
