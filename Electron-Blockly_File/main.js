import { app, BrowserWindow, ipcMain,dialog } from 'electron'
import path  from 'path'
const fs =await import('fs');
function createWindow() {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false, // consider changing for production for security reasons
            enableRemoteModule: true // if you need remote modul
        }
    });

    win.loadFile('./renderer/index.html');
}

app.whenReady().then(() => {
    createWindow();

    ipcMain.on('save-code-to-file', async (event, code) => {
        const { filePath } = await dialog.showSaveDialog({
            buttonLabel: 'Save Code',
            filters: [{ name: 'Text Files', extensions: ['txt'] },{ name: 'Python Files', extensions: ['py'] }]
        });

        if (filePath) {
            fs.appendFileSync(filePath, code);
            event.reply('code-save-status', 'File saved successfully to ' + filePath);
        } else {
            event.reply('code-save-status', 'Save cancelled');
        }
    });
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});