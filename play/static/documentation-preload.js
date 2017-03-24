// So we can do IPC without requiring node integration.
var electron = require('electron');
window.ipcRenderer = electron.ipcRenderer;

var webFrame = electron.webFrame;
ipcRenderer.on('menu:zoom-in', (event, arg) => {
  webFrame.setZoomLevel(webFrame.getZoomLevel() + 1);
});

ipcRenderer.on('menu:zoom-out', (event, arg) => {
  webFrame.setZoomLevel(webFrame.getZoomLevel() - 1);
});

ipcRenderer.on('menu:zoom-reset', (event, arg) => {
  webFrame.setZoomLevel(0);
});
