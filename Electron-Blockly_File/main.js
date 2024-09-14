import { app, BrowserWindow, ipcMain, dialog, session } from 'electron';
import path from 'path';
import fs from 'fs';
import readline from 'readline';


const filePathTraining = './generatedCode/Error.py';
const filePaths = [];

let currentPath = filePathTraining;

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
        const filePath = path.join('./projectsrc/projectskeleton2.py');
        console.log("READING FILE");

        fs.readFile(filePath, 'utf-8', (err, data) => {
            console.log("reading...");
            if (err) {
                console.error('Failed to load file:', err);
                return;
            }


        });
    }).catch(err => {
        console.error('Error clearing storage data:', err);
        // Still create the window even if there was an error clearing data
        createWindow();
    });


});

function insertCommentAtLine(filePath, viewName, lineIndex) {
    try {
        // Read the current content of the file
        let content = fs.readFileSync(filePath, 'utf8');

        // Split the content into an array of lines
        let lines = content.split('\n');

        // Create the comment to be inserted
        const comment = `#${viewName}+\n#${viewName}-`;

        // Insert the comment at the specified line index
        if (lineIndex >= 0 && lineIndex <= lines.length) {
            lines.splice(lineIndex, 0, comment);  // Insert the comment at the given index
        } else {
            console.error("Line index out of bounds");
            return;
        }

        // Join the lines back together and write the modified content back to the file
        const updatedContent = lines.join('\n');
        fs.writeFileSync(filePath, updatedContent, 'utf8');

        console.log(`Comment successfully inserted at line ${lineIndex}`);
    } catch (err) {
        console.error('Error while modifying the file:', err);
    }
}


ipcMain.on('save-code-to-file', async (event, code) => {
    fs.writeFileSync(currentPath, code);
});

ipcMain.on('save-block-to-file', async (event, codeArray) => {
    for (let i = 0; i < currentPath.length; i++) {
        console.log("Saving " + codeArray[i] + " to "  + filePaths[i])
        fs.writeFileSync(filePaths[i], codeArray[i]);
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
    const filePath = `./generatedCode/${viewName}.py`;
    filePaths.push(filePath);
    fs.writeFileSync(filePath, '', 'utf8');
    event.sender.send('new-view-created', { message: `File created: ${viewName}`, viewName: viewName });
});

ipcMain.on('create-new-view-with-index', (event, viewName, index) => {
    const filePath = `./generatedCode/${viewName}.py`;
    filePaths.push(filePath);
    fs.writeFileSync(filePath, '', 'utf8');
    const filePath2 = './projectsrc/projectskeleton2.py';
    insertCommentAtLine(filePath2, viewName, index);
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
