const { contextBridge, ipcRenderer } = require('electron');

window.addEventListener('DOMContentLoaded', () => {
    console.log('Preload script running');

    contextBridge.exposeInMainWorld('electronAPI', {
        saveCodeToFile: (data) => ipcRenderer.send('save-code-to-file', data),
        receiveCodeSaveStatus: (callback) => ipcRenderer.on('code-save-status', (event, ...args) => callback(...args))
    });
});
