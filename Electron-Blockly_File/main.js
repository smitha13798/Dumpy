import { app, BrowserWindow, ipcMain,dialog } from 'electron'

import path  from 'path'
const fs =await import('fs');
function createWindow() {
    // Clear the cache before creating the window
        console.log('Cache cleared!');
        const win = new BrowserWindow({
            width: 800,
            height: 600,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false,  // consider changing for production for security reasons
                enableRemoteModule: true  // if you need remote module
            }
        });

        win.loadFile('./renderer/index.html');
}

app.whenReady().then(() => {
    createWindow();
    var filePathModel = '/Users/tobias/Documents/Dumpy/Electron-Blockly_File/generatedCode/modelDefinition.py'
    var filePathData = '/Users/tobias/Documents/Dumpy/Electron-Blockly_File/generatedCode/dataloaderDefinition.py'
    var currentPath = filePathModel
    ipcMain.on('save-code-to-file', async (event, code) => {
        /*const { filePath } = await dialog.showSaveDialog({
            buttonLabel: 'Save Codes',
            filters: [{ name: 'Text Files', extensions: ['txt'] },{ name: 'Python Files', extensions: ['py'] }]
        });*/
        fs.writeFileSync(currentPath, code);


    });


    ipcMain.on('change-view', () => {
        if(currentPath===filePathModel){
            currentPath = filePathData;
            return;
        }
        currentPath = filePathModel;
    })
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
