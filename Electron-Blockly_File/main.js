import { app, BrowserWindow, ipcMain,dialog } from 'electron'

const fs =await import('fs');
var filePathModel = '/Users/tobias/Documents/Dumpy/Electron-Blockly_File/generatedCode/modelDefinition.py'
var filePathData = '/Users/tobias/Documents/Dumpy/Electron-Blockly_File/generatedCode/dataloaderDefinition.py'
var filePathTraining = ' generatedCode/trainingDefinition.py'
var filePaths = [];
filePaths[0] = filePathModel;
filePaths[1] = filePathData;
filePaths[2] = filePathTraining;

var currentPath = filePathModel
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

    ipcMain.on('change-view-option', async (event, view) => {
        /*const { filePath } = await dialog.showSaveDialog({
            buttonLabel: 'Save Codes',
            filters: [{ name: 'Text Files', extensions: ['txt'] },{ name: 'Python Files', extensions: ['py'] }]
        });*/
        console.log("Changing frm main to" + view)
        currentPath = filePaths[view];
        console.log("current Path is.." + currentPath)
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



