import { Menu, dialog, app, ipcMain as ipc, BrowserWindow } from 'electron';
import { resolve, join } from 'path';
import { existsSync } from 'fs';

import Rx from 'rxjs/Rx';

import {
  launchNotebookFromFile,
  launchNewNotebook,
} from './notebook-window';

import { defaultMenu } from './menu';

import prepareEnv from './prepare-env';

const log = require('electron-log');

const path = require('path');

const argv = require('yargs')
  .version()
  .usage('Usage: nao-play <notebooks> [options]')
  .example('nao-play notebook1.ipynb notebook2.ipynb', 'Open notebooks')
  .example('nao-play --tensorboard', 'Launch a TensorBoard')
  .describe('tensorboard', 'Launch a TensorBoard')
  .alias('v', 'version')
  .alias('h', 'help')
  .describe('verbose', 'Display debug information')
  .help('help')
  .parse(process.argv.slice(1));

log.info('args', argv);

const notebooks = argv._
  .filter(x => /(.naonb)$/.test(x))
  .filter(x => existsSync(resolve(x)));

ipc.on('new-kernel', (event, k) => {
  launchNewNotebook(k);
});

ipc.on('open-notebook', (event, filename) => {
  launchNotebookFromFile(resolve(filename));
});

const electronReady$ = Rx.Observable.fromEvent(app, 'ready');

const fullAppReady$ = Rx.Observable.zip(
  electronReady$,
  prepareEnv
).first();

function closeAppOnNonDarwin() {
  // On macOS, we want to keep the app and menu bar active
  if (process.platform !== 'darwin') {
    app.quit();
  }
}
const windowAllClosed = Rx.Observable.fromEvent(app, 'window-all-closed');
windowAllClosed
  .skipUntil(fullAppReady$)
  .subscribe(closeAppOnNonDarwin);

const openFile$ = Rx.Observable.fromEvent(
  app,
  'open-file', (event, filename) => ({ event, filename })
);

function openFileFromEvent({ event, filename }) {
  event.preventDefault();
  launchNotebookFromFile(resolve(filename));
}

// Since we can't launch until app is ready
// and macOS will send the open-file events early,
// buffer those that come early.
openFile$
  .buffer(fullAppReady$) // Form an array of open-file events from before app-ready
  .first() // Should only be the first
  .subscribe((buffer) => {
    // Now we can choose whether to open the default notebook
    // based on if arguments went through argv or through open-file events
    if (notebooks.length <= 0 && buffer.length <= 0) {
      log.info('launching an empty notebook by default. Actually, not doing that.');
      launchNewNotebook();
    } else {
      notebooks
        .forEach((f) => {
          try {
            launchNotebookFromFile(resolve(f));
          } catch (e) {
            log.error(e);
            console.error(e);
          }
        });
    }
    buffer.forEach(openFileFromEvent);
  });

// All open file events after app is ready
openFile$
  .skipUntil(fullAppReady$)
  .subscribe(openFileFromEvent);

fullAppReady$
  .subscribe(() => {
    Menu.setApplicationMenu(defaultMenu);
  });
