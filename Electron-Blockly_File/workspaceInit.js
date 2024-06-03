

console.log('Attempting to load toolbox module...');
const toolboxPath = './toolbox.js';
const blocksPath = './blocks.js';
const serializerPath = './serialization.js';
const javascriptPath = './generator/javascript.js';
let ws;
document.addEventListener('DOMContentLoaded', async function () {
    let toolbox;
    let blocks;
    let load;
    let save;
    let forBlock;
    try {
        const module = await import(toolboxPath);
        toolbox = module.toolbox;
        console.log('Toolbox loaded:', toolbox);
    } catch (error) {
        console.error('Error loading toolbox:', error);
    }
    try {
        const module = await import(blocksPath);
        blocks = module.blocks;
    }
    catch (error) {
        console.error('Error loading toolbox:', error);
    }
    try {
        const module = await import(serializerPath);
        load = module.load;
        save = module.save;
    }
    catch (error) {
        console.error('Error loading toolbox:', error);
    }
    try {
            const module = await import(javascriptPath);
            forBlock = module.forBlock;
    }
    catch (error) {
        console.error('Error loading toolbox:', error);
    }

    const Blockly = require('blockly');
    const pythonGenerator = require('blockly/python');
    const javaScriptGenerator = require('blockly/javascript');
    Blockly.common.defineBlocks(blocks)
    Object.assign(javaScriptGenerator.javascriptGenerator.forBlock,forBlock);
    const blocklyDiv = document.getElementById('blocklyDiv');
    if (blocklyDiv) {
        console.log('blocklyDiv found:', blocklyDiv);
        try {
            ws = Blockly.inject(blocklyDiv, {toolbox});
            console.log('Blockly has been initialized', ws);
        } catch (error) {
            console.error('Failed to initialize Blockly:', error);
        }
    } else {
        console.log('Failed to find blocklyDiv');
    }




    const runCode = () => {
        const code = pythonGenerator.pythonGenerator.workspaceToCode(ws)

        console.log("COde comes here"+ code)
        //eval(code); Checking for JS code which wont work
    };



// Load the initial state from storage and run the code.
    load(ws);
    runCode();

// Every time the workspace changes state, save the changes to storage.
    ws.addChangeListener((e) => {
        // UI events are things like scrolling, zooming, etc.
        // No need to save after one of these.
        console.log("Changed...")
        if (e.isUiEvent) return;
        save(ws);
    });


// Whenever the workspace changes meaningfully, run the code again.
    ws.addChangeListener((e) => {
        // Don't run the code when the workspace finishes loading; we're
        // already running it once when the application starts.
        // Don't run the code during drags; we might have invalid state.
        if (e.isUiEvent || e.type == Blockly.Events.FINISHED_LOADING ||
            ws.isDragging()) {
            console.log("Dragging...")
            return;
        }

        runCode();
    });

    const supportedEvents = new Set([
        Blockly.Events.BLOCK_CHANGE,
        Blockly.Events.BLOCK_CREATE,
        Blockly.Events.BLOCK_DELETE,
        Blockly.Events.BLOCK_MOVE,
    ]);

    function updateCode(event) {
        if (ws.isDragging()) return; // Don't update while changes are happening.
        if (!supportedEvents.has(event.type)) return;

        const code = pythonGenerator.pythonGenerator.workspaceToCode(ws);
        console.log("changed")
        document.getElementById('textarea').textContent = code;
    }

    ws.addChangeListener(updateCode);
});
const { ipcRenderer } = require('electron');

document.addEventListener('DOMContentLoaded', function() {
    const saveButton = document.getElementById('saveButton');
    saveButton.addEventListener('click', function() {
        const code = document.getElementById('textarea').textContent;
        ipcRenderer.send('save-code-to-file', code)
            .then(filePath => {
                console.log('File saved:', filePath);
            })
            .catch(error => {
                console.error('Error saving file:', error);
            });
    });
});




