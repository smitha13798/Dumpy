const { contextBridge, ipcRenderer } = require('electron');

window.addEventListener('DOMContentLoaded', () => {
    contextBridge.exposeInMainWorld('electronAPI', {
        saveCodeToFile: (data) => ipcRenderer.send('save-code-to-file', data),
        receiveCodeSaveStatus: (callback) => ipcRenderer.on('code-save-status', (event, ...args) => callback(...args)),
        createNewView: (viewName) => ipcRenderer.invoke('createNewView', viewName)
    });
});




