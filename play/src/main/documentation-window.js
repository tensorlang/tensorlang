/* @flow */
import path from 'path';

import { shell, BrowserWindow } from 'electron';

export function deferURL(event, url) {
  event.preventDefault();
  if (!url.startsWith('file:')) {
    shell.openExternal(url);
  }
}

const iconPath = path.join(__dirname, '..', '..', 'static', 'icon.png');

const index = path.join(__dirname, '..', '..', '..', 'doc', 'build', 'index.html');
const preload = path.join(__dirname, '..', '..', 'static', 'documentation-preload.js');

export function launchDocumentation() {
  let win = new BrowserWindow({
    width: 800,
    height: 1000,
    icon: iconPath,
    title: `Nao Documentation`,
    webPreferences: {
      preload: preload,
      // Required to avoid weird errors
      nodeIntegration: false,
    }
  });

  win.loadURL(`file://${index}`);
  win.webContents.on('will-navigate', deferURL);

  win.on('page-title-updated', function(event) {
    event.preventDefault();
  });

  // Emitted when the window is closed.
  win.on('closed', () => {
    win = null;
  });

  return win;
}
