import { app, BrowserWindow, ipcMain, dialog } from 'electron';
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
    createWindow();

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
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
