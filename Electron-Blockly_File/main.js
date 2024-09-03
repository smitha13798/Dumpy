import { app, BrowserWindow, ipcMain, dialog, session } from 'electron';
import path from 'path';
import fs from 'fs';
import readline from 'readline';

const filePathModel = './generatedCode/modelDefinition.py';
const filePathData = './generatedCode/dataloaderDefinition.py';
const filePathTraining = './generatedCode/trainingDefinition.py';
const filePaths = [filePathModel, filePathData, filePathTraining, './generatedCode/decoderDefinition.py'];

let currentPath = filePathModel;

function createWindow() {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false, // Consider enabling context isolation in production for security reasons
        },
    });

    win.loadFile('./renderer/index.html');
}

app.whenReady().then(() => {
    // Clear cache, cookies, storage, and other session data before creating the window
    const ses = session.defaultSession;

    // Clear cache
    ses.clearCache().then(() => {
        console.log('Cache cleared');
    });

    // Clear cookies
    ses.clearStorageData({ storages: ['cookies'] }).then(() => {
        console.log('Cookies cleared');
    });

    // Clear local storage, session storage, indexedDB, and other storage data
    ses.clearStorageData({
        storages: ['localstorage', 'sessions', 'indexdb', 'websql', 'serviceworkers', 'cachestorage'],
    }).then(() => {
        console.log('All session storage data cleared');

        // Now create the window
        createWindow();
    }).catch(err => {
        console.error('Error clearing storage data:', err);
        // Still create the window even if there was an error clearing data
        createWindow();
    });
});

ipcMain.on('save-code-to-file', async (event, code) => {
    fs.writeFileSync(currentPath, code);
});

ipcMain.on('change-view', (event) => {
    if (currentPath === filePathModel) {
        currentPath = filePathData;
    } else {
        currentPath = filePathModel;
    }
});

ipcMain.on('read-outputjson', (event) => {
    const functionNamesContent = fs.readFileSync('./output.json', 'utf8');
    const functionNamesJson = JSON.parse(functionNamesContent);
    console.log("Are in server now")
    event.sender.send('response-outputjson', functionNamesJson);
});

ipcMain.on('read-file', (event) => {
    const filePath = 'projectsrc/CodeToBlockDemo.py';
    const readInterface = readline.createInterface({
        input: fs.createReadStream(filePath),
        output: process.stdout,
        console: false,
    });

    readInterface.on('line', (line) => {
        console.log('Reading ..' + line);
    });
});

ipcMain.on('change-view-option', (event, view) => {
    currentPath = filePaths[view];
    console.log('Current Path is:', currentPath);
});

ipcMain.on('create-new-view', (event, viewName, index) => {
    const filePath = `./generatedCode/${viewName}${index}.py`;
    filePaths.push(filePath);
    fs.writeFileSync(filePath, '', 'utf8');
    event.sender.send('new-view-created', { message: `File created: ${viewName}`, viewName: viewName });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
