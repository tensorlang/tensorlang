/* @flow */
import { dialog, app, shell, Menu, ipcMain as ipc,
         BrowserWindow } from 'electron';
import * as path from 'path';
import { launchNotebookFromFile, launchNewNotebook } from './notebook-window';
import { openTensorBoard } from './tensorboard-window';
import { launchDocumentation } from './documentation-window';
import { launchTraceViewerForFile } from './trace-viewer-window';
import { installShellCommand } from './cli';

function getExampleNotebooksDir() {
  if (process.env.NODE_ENV === 'development') {
    return path.resolve(path.join(__dirname, '..', '..', 'example-notebooks'));
  }
  return path.join(process.resourcesPath, 'example-notebooks');
}

const exampleNotebooksDirectory = getExampleNotebooksDir();

function send(focusedWindow, eventName, obj) {
  if (!focusedWindow) {
    console.error('renderer window not in focus (are your devtools open?)');
    return;
  }
  focusedWindow.webContents.send(eventName, obj);
}

function createSender(eventName, obj) {
  return (item, focusedWindow) => {
    send(focusedWindow, eventName, obj);
  };
}

const file = {
  label: '&File',
  submenu: [
    {
      label: `&New Notebook`,
      click: () => launchNewNotebook(),
      accelerator: 'CmdOrCtrl+N',
    },
    {
      label: '&Open Notebook...',
      click: () => {
        dialog.showOpenDialog(
          {
            title: 'Open Notebook...',
            defaultPath: (process.cwd() === '/') ? app.getPath('home') : undefined,
            filters: [
              { name: 'Notebooks', extensions: ['ipynb'] },
            ],
            properties: [
              'openFile',
            ],
          },
          (fname) => {
            if (fname) {
              launchNotebookFromFile(fname[0]);
            }
          }
        );
      },
      accelerator: 'CmdOrCtrl+O',
    },
    {
      label: 'Open &TensorBoard...',
      click: () => {
        dialog.showOpenDialog(
          {
            title: 'Open TensorBoard...',
            properties: [
              'openDirectory',
              'showHiddenFiles',
            ],
          },
          (fname) => {
            if (fname) {
              openTensorBoard(fname[0]);
            }
          }
        );
      },
      accelerator: 'CmdOrCtrl+Shift+O',
    },
    {
      label: '&Open Trace...',
      click: () => {
        dialog.showOpenDialog(
          {
            title: 'Open Trace...',
            defaultPath: (process.cwd() === '/') ? app.getPath('home') : undefined,
            filters: [
              { name: 'Trace File', extensions: ['trace', 'json'] },
            ],
            properties: [
              'openFile',
              'showHiddenFiles',
            ],
          },
          (fname) => {
            if (fname) {
              launchTraceViewerForFile(fname[0]);
            }
          }
        );
      },
      accelerator: 'CmdOrCtrl+Alt+O',
    },
    {
      type: 'separator',
    },
    {
      label: '&Save Notebook',
      click: createSender('menu:save'),
      accelerator: 'CmdOrCtrl+S',
    },
    {
      label: 'Save Notebook &As...',
      click: (item, focusedWindow) => {
        const opts = {
          title: 'Save Notebook As...',
          filters: [{ name: 'Notebooks', extensions: ['ipynb'] }],
        };

        if (process.cwd() === '/') {
          opts.defaultPath = app.getPath('home');
        }

        dialog.showSaveDialog(opts, (filename) => {
          if (!filename) {
            return;
          }

          const ext = path.extname(filename) === '' ? '.ipynb' : '';
          send(focusedWindow, 'menu:save-as', `${filename}${ext}`);
        });
      },
      accelerator: 'CmdOrCtrl+Shift+S',
    },
    {
      type: 'separator',
    },
    {
      label: '&Publish Notebook as Gist',
      click: createSender('menu:publish:gist'),
    },
    {
      label: '&Export Notebook as PDF',
      click: createSender('menu:exportPDF'),
      accelerator: 'CmdOrCtrl+Shift+E',
    },
  ],
};

export const edit = {
  label: 'Edit',
  submenu: [
    {
      label: 'Cut',
      accelerator: 'CmdOrCtrl+X',
      role: 'cut',
    },
    {
      label: 'Copy',
      accelerator: 'CmdOrCtrl+C',
      role: 'copy',
    },
    {
      label: 'Paste',
      accelerator: 'CmdOrCtrl+V',
      role: 'paste',
    },
    {
      label: 'Select All',
      accelerator: 'CmdOrCtrl+A',
      role: 'selectall',
    },
    {
      type: 'separator',
    },
    {
      label: 'New Code Cell',
      accelerator: 'CmdOrCtrl+Shift+N',
      click: createSender('menu:new-code-cell'),
    },
    {
      label: 'New Text Cell',
      accelerator: 'CmdOrCtrl+Shift+M',
      click: createSender('menu:new-text-cell'),
    },
    {
      label: 'Copy Cell',
      accelerator: 'CmdOrCtrl+Shift+C',
      click: createSender('menu:copy-cell'),
    },
    {
      label: 'Cut Cell',
      accelerator: 'CmdOrCtrl+Shift+X',
      click: createSender('menu:cut-cell'),
    },
    {
      label: 'Paste Cell',
      accelerator: 'CmdOrCtrl+Shift+V',
      click: createSender('menu:paste-cell'),
    },
    {
      type: 'separator'
    },
  ],
};

export const view = {
  label: 'View',
  submenu: [
    {
      label: 'Toggle Full Screen',
      accelerator: (() => {
        if (process.platform === 'darwin') {
          return 'Ctrl+Command+F';
        }
        return 'F11';
      })(),
      click: (item, focusedWindow) => {
        if (focusedWindow) {
          focusedWindow.setFullScreen(!focusedWindow.isFullScreen());
        }
      },
    },
    {
      label: 'Developer',
      submenu: [
        {
          label: 'Reload Window',
          accelerator: (() => {
            if (process.platform === 'darwin') {
              return 'Ctrl+Alt+Command+L';
            }
            return 'Alt+Ctrl+Shift+L';
          })(),
          click: (item, focusedWindow) => {
            if (focusedWindow) {
              focusedWindow.reload();
            }
          },
        },
        ...((process.env.NODE_ENV !== 'development') ? [] : [{
          label: 'Install React Developer Tools',
          click: (item, focusedWindow) => {
            var rdev = require('electron-react-devtools');
            rdev.install();
          }
        }]),
        {
          label: 'Toggle Developer Tools',
          accelerator: (() => {
            if (process.platform === 'darwin') {
              return 'Alt+Command+I';
            }
            return 'Ctrl+Shift+I';
          })(),
          click: (item, focusedWindow) => {
            if (focusedWindow) {
              focusedWindow.toggleDevTools();
            }
          },
        },
      ]
    },
    {
      type: 'separator'
    },
    {
      label: 'Zoom In',
      accelerator: 'CmdOrCtrl+=',
      click: createSender('menu:zoom-in'),
    },
    {
      label: 'Zoom Out',
      accelerator: 'CmdOrCtrl+-',
      click: createSender('menu:zoom-out'),
    },
    {
      label: 'Actual Size',
      accelerator: 'CmdOrCtrl+0',
      click: createSender('menu:zoom-reset'),
    },
    {
      type: 'separator'
    },
  ],
};

export const control = {
  label: 'Control',
  submenu: [
    {
      label: 'Run All',
      click: createSender('menu:run-all'),
      accelerator: 'CmdOrCtrl+R'
    },
    {
      label: 'Run All Below',
      click: createSender('menu:run-all-below'),
      accelerator: 'CmdOrCtrl+Shift+R'
    },
    {
      label: 'Clear All Outputs',
      click: createSender('menu:clear-all'),
    },
    {
      label: 'Unhide All Outputs',
      click: createSender('menu:unhide-all'),
    },
    {
      type: 'separator'
    },
    {
      label: 'Kernel',
      submenu: [
        {
          label: '&Kill Running Kernel',
          click: createSender('menu:kill-kernel'),
        },
        {
          label: '&Interrupt Running Kernel',
          click: createSender('menu:interrupt-kernel'),
        },
        {
          label: 'Restart Running Kernel',
          click: createSender('menu:restart-kernel'),
        },
        {
          label: 'Restart and Clear All Cells',
          click: createSender('menu:restart-and-clear-all'),
        },
      ]
    },
  ],
};

const windowDraft = {
  label: 'Window',
  role: 'window',
  submenu: [
    {
      label: 'Minimize',
      accelerator: 'CmdOrCtrl+M',
      role: 'minimize',
    },
    {
      label: 'Close',
      accelerator: 'CmdOrCtrl+W',
      role: 'close',
    },
  ],
};

if (process.platform === 'darwin') {
  windowDraft.submenu.push(
    {
      type: 'separator',
    },
    {
      label: 'Bring All to Front',
      role: 'front',
    }
  );
}

export const window = windowDraft;

const shellCommands = {
  label: 'Install Shell Commands',
  click: () => installShellCommand(),
};

export const help = {
  label: 'Help',
  role: 'help',
  submenu: [
    {
      label: 'Open Documentation',
      click: () => { launchDocumentation(); }
    },
    {
      label: '&Open Example Notebook',
      click: launchNotebookFromFile.bind(null, path.join(exampleNotebooksDirectory, 'intro.ipynb')),
    },
    {
      type: 'separator',
    },
    {
      label: 'Report Issue',
      click: () => { shell.openExternal('https://github.com/ajbouh/nao/issues/new'); }
    },
    {
      label: 'Search Issues',
      click: () => { shell.openExternal('https://github.com/ajbouh/nao/issues/'); }
    },
    {
      type: 'separator',
    },
    {
      label: 'Nao Home',
      click: () => { shell.openExternal('https://github.com/ajbouh/nao'); }
    },
  ]
};

const name = 'Nao Play';
app.setName(name);

const icon = path.join(__dirname, '..', '..', 'build', 'icon.png');
app.dock.setIcon(icon);

export const named = {
  label: name,
  submenu: [
    {
      label: `About ${name}`,
      role: 'about',
    },
    {
      type: 'separator',
    },
    shellCommands,
    {
      type: 'separator',
    },
    {
      label: 'Services',
      role: 'services',
      submenu: [],
    },
    {
      type: 'separator',
    },
    {
      label: `Hide ${name}`,
      accelerator: 'Command+H',
      role: 'hide',
    },
    {
      label: 'Hide Others',
      accelerator: 'Command+Alt+H',
      role: 'hideothers',
    },
    {
      label: 'Show All',
      role: 'unhide',
    },
    {
      type: 'separator',
    },
    {
      label: 'Quit',
      accelerator: 'Command+Q',
      click: () => app.quit(),
    },
  ],
};

const template = [];

if (process.platform === 'darwin') {
  template.push(named);
}

template.push(file);
template.push(edit);
template.push(view);
template.push(control);

// Application specific functionality should go before window and help
template.push(window);
template.push(help);

export const defaultMenu = Menu.buildFromTemplate(template);
