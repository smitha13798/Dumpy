

console.log('Attempting to load toolbox module...');
const toolboxPath = './toolbox.js';
const blocksPath = './blocks.js';
const serializerPath = './serialization.js';
const javascriptPath = './generator/javascript.js';
const Blockly = require('blockly');
const pythonGenerator = require('blockly/python');
const javaScriptGenerator = require('blockly/javascript');
let ws;
const WorkspaceStates = [];
let currentWS = 0;
const { ipcRenderer } = require('electron');
const fs = require('fs');
const path = require('path');
// Utility function to add sub-options to a dropdown
var editor = ace.edit("codeeditor");
editor.setTheme("ace/theme/monokai");
editor.session.setMode("ace/mode/python");

// Function to create and return a block with configured input fields


// Function to connect a variable block to a function block
function appendBlockToClass(classBlock, blockToAppend) {
    // Get the 'METHODS' input from the class block
    const methodsInput = classBlock.getInput('METHODS');

    // Check if the 'METHODS' input and its connection exist
    if (methodsInput && methodsInput.connection) {
        // Get the current connected block (the last method in the class)
        let lastMethodBlock = methodsInput.connection.targetBlock();

        if (lastMethodBlock) {
            // Traverse to the last connected method in the chain
            while (lastMethodBlock.nextConnection && lastMethodBlock.nextConnection.isConnected()) {
                lastMethodBlock = lastMethodBlock.nextConnection.targetBlock();
            }

            // Connect the new blockToAppend to the next connection of the last method
            lastMethodBlock.nextConnection.connect(blockToAppend.previousConnection);
        } else {
            // If no methods are connected, directly connect the blockToAppend
            methodsInput.connection.connect(blockToAppend.previousConnection);
        }
    } else {
        console.error("No valid 'METHODS' input or connection found in the class block.");
    }
}

// Function to append a block to a function block
function appendBlockToFunction(functionBlock, blockToAppend) {
    const methodsInput = functionBlock.getInput('METHODS');

    if (methodsInput?.connection && blockToAppend.previousConnection) {
        methodsInput.connection.connect(blockToAppend.previousConnection);
    } else {
        console.error("No valid 'METHODS' input or connection found in the function block.");
    }
}

let lastBlock = null;

// Function to append a block to the workspace and manage connections
function appendBlockToWorkspace(blockInfo) {
    // Create and configure the new block
    let block = ws.newBlock(blockInfo.name);
    let variableBlock = null;

    // If there's an assignment, create a variable block
    if (blockInfo.assigned) {
        console.log("Creating variable assignment block");
        variableBlock = ws.newBlock('set_var');
        variableBlock.setFieldValue(blockInfo.assigned, 'SET_VARIABLE');
        variableBlock.initSvg();
        variableBlock.render();
    }
    let inputValue = blockInfo.parameters.toString();
    if (inputValue === "()") inputValue = "";

    // Configure the new block with its parameters
    block.inputList.forEach(input => {
        if (input.fieldRow) {
            input.fieldRow.forEach(field => {
                if (field.name !== undefined) {

                    block.setFieldValue(inputValue, field.name);
                    inputValue="";
                }
            });
        }

    });

    // Initialize and render the new block
    block.initSvg();
    block.render();

    // Debugging: Log connection states


    if (variableBlock) {
        // Ensure that 'VALUE' input exists and has a connection
        const valueInput = variableBlock.getInput('VALUE');
        const valueInputConnection = valueInput ? valueInput.connection : null;
        const functionOutputConnection = block.outputConnection;

        // Debugging: Check if connections exist
        console.log('Value Input Connection:', valueInputConnection);
        console.log('Function Output Connection:', functionOutputConnection);

        if (valueInputConnection && functionOutputConnection) {
            // Connect the variable block to the function block
            valueInputConnection.connect(functionOutputConnection);
            block = variableBlock;
        } else {
            console.error('One or both connections are not available for variable block.');
        }

    }

    // Connect to the previous lastBlock if applicable
    if (lastBlock) {
        // Ensure that connections are valid before attempting to connect
        const lastBlockConnection = lastBlock.nextConnection;
        const currentBlockConnection = block.previousConnection;
        console.log("CONNECTING BLOCK TO BLOCK")
        // Debugging: Check if connections exist
        console.log('Last Block Connection:', lastBlockConnection);
        console.log('Current Block Connection:', currentBlockConnection);

        if (lastBlockConnection && currentBlockConnection) {
            lastBlockConnection.connect(currentBlockConnection);
        } else {
            console.error('Connection points are not available for lastBlock.');
        }
    }

    // Update the lastBlock reference
    lastBlock = block;
    if(variableBlock){
        lastBlock = variableBlock;
    }
    variableBlock = null;
    // Force a workspace update
    ws.resizeContents();
}



function getMaxValue() {

    const selectElement = document.getElementById('ViewList');
    return Array.from(selectElement.options).reduce((max, option) => Math.max(max, parseFloat(option.value)), -Infinity);
}

function getBlockInformation(e) {
    return {
        assigned: e.assigned,
        name: e.function,
        parameters: e.parameters
    };
}
document.addEventListener('DOMContentLoaded', function () {
    var editor = ace.edit("codeeditor");
    editor.setTheme("ace/theme/monokai");
    editor.session.setMode("ace/mode/python");

    document.getElementById('loadFileButton').addEventListener('click', function() {
        const filePath = path.join('./projectsrc/projectskeleton2.py'); //Reads the generated file
        console.log("READING FILE")
        fs.readFile(filePath, 'utf-8', (err, data) => {
            console.log("reading...")
            if (err) {
                console.error('Failed to load file:', err);
                return;
            }
            editor.setValue(data, -1);
        });
    });

});




document.addEventListener('DOMContentLoaded', function () {
    const createNewViewButton = document.getElementById('createNewView');
    let buttonState = 0;
    let viewName = "";
    let select = document.getElementById('ViewList');
    createNewViewButton.addEventListener('click', () => {
        if (buttonState === 0) {
            viewName = document.getElementById('new-view-name').value.trim();
            document.getElementById('new-view-name').value = "";
            document.getElementById('new-view-name').placeholder = 'Index...';
            buttonState = 1;
        } else {
            const index = document.getElementById('new-view-name').value.trim();





            buttonState = 0;
            const workspaceState = Blockly.serialization.workspaces.save(ws);
            WorkspaceStates[select.length-1] = JSON.stringify(workspaceState);
            var value = 0;
            if(select.length!==0){
                value=select.length-1;
            }

            select.add(new Option(`${viewName}${index}`,value))
            currentWS  =value;
            ws.clear();
            select.value  = currentWS;
            ipcRenderer.send('create-new-view', `${viewName}`,index);

        }
    });

    const swapButton = document.getElementById('swap');
    swapButton.addEventListener('click', function () {
        const code = document.getElementById('textarea').textContent;
        ipcRenderer.send('save-code-to-file', code);
        ws.clear();
        ipcRenderer.send('change-view-option');
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const searchBox = document.getElementById('searchBox');
    const searchResults = document.getElementById('searchResults');

    // Listen for input events in the search box
    searchBox.addEventListener('input', function () {



        const searchTerm = searchBox.value.toLowerCase();
        searchResults.innerHTML = '';  // Clear previous results

        // Get the toolbox object
        const toolbox = ws.getToolbox();
        const categories = toolbox.getToolboxItems();

        let foundBlocks = false;

        // Loop through all categories
        categories.forEach(function (category) {
            const categoryName = category.name_;  // Get category name
            const categoryBlocks = category.getContents();  // Get blocks in the category

            // Loop through blocks within the category
            categoryBlocks.forEach(function (block) {
                if (block.kind === 'block') {
                    const blockType = block.type;  // Get the block type (name)

                    // Check if the block type includes the search term
                    if (blockType.toLowerCase().includes(searchTerm) && searchTerm!=="") {
                        foundBlocks = true;

                        // Display the block's category and block name
                        searchResults.innerHTML += `<p>Block: <strong>${blockType}</strong> is in Category: <strong>${categoryName}</strong></p>`;
                    }
                }
            });
        });

        // If no blocks were found, display a message
        if (!foundBlocks && searchTerm.length > 0) {
            searchResults.innerHTML = `<p>No blocks found matching "${searchTerm}"</p>`;
        }
    });
});

document.addEventListener('DOMContentLoaded', async function () {
    let toolbox, blocks, load, save, forBlock;

    try {
        toolbox = (await import(toolboxPath)).toolbox;
        console.log('Toolbox loaded:', toolbox);
    } catch (error) {
        console.error('Error loading toolbox:', error);
    }

    try {
        blocks = (await import(blocksPath)).blocks;
    } catch (error) {
        console.error('Error loading blocks:', error);
    }

    try {
        ({ load, save } = await import(serializerPath));
    } catch (error) {
        console.error('Error loading serializer:', error);
    }

    try {
        forBlock = (await import(javascriptPath)).forBlock;
    } catch (error) {
        console.error('Error loading JavaScript generator:', error);
    }

    Blockly.common.defineBlocks(blocks);
    Object.assign(javaScriptGenerator.javascriptGenerator.forBlock, forBlock);

    const blocklyDiv = document.getElementById('blocklyDiv');
    if (blocklyDiv) {
        console.log('blocklyDiv found:', blocklyDiv);
        ws = Blockly.inject(blocklyDiv, { toolbox });
        console.log('Blockly has been initialized', ws);
    } else {
        console.error('Failed to find blocklyDiv');
    }

    const runCode = () => {
        const code = pythonGenerator.pythonGenerator.workspaceToCode(ws);
    };

    load(ws); // Load the initial state for the workspace
    runCode();

    ws.addChangeListener(e => {
        if (!e.isUiEvent && e.type !== Blockly.Events.FINISHED_LOADING && !ws.isDragging()) {
            runCode();
            document.getElementById('textarea').textContent = pythonGenerator.pythonGenerator.workspaceToCode(ws);
        }
    });

    const saveButton = document.getElementById('saveButton');
    saveButton.onclick = () => {
        ipcRenderer.send('save-code-to-file', pythonGenerator.pythonGenerator.workspaceToCode(ws));
    };

    const viewList = document.getElementById('ViewList');
    viewList.onchange = function () {
        const selectedIndex = viewList.value;

        // Save current workspace state before switching
        const workspaceState = Blockly.serialization.workspaces.save(ws);

        WorkspaceStates[currentWS] = JSON.stringify(workspaceState);
        // Update current workspace index
        currentWS = selectedIndex;

        // Clear the workspace before loading the new state


        // Notify backend to change the view option
        ipcRenderer.send('change-view-option', selectedIndex);

        // Check if the state exists for the selected view, then load it
        if (WorkspaceStates[selectedIndex]) {
            console.log("LOADING NEW STATE")
            const workspaceStateString = WorkspaceStates[selectedIndex];

// Parse the state string back to a JSON object
            const workspaceState = JSON.parse(workspaceStateString);

// Clear the current workspace

// Load the workspace state
            Blockly.serialization.workspaces.load(workspaceState, ws);

// Resize the workspace to fit the restored content
            ws.resizeContents();
        }

        // Run any additional initialization code (like code generation)
        runCode();

        // Force a full workspace refresh
        ws.resizeContents();

        // Handle sub-options display logic (optional)



    };


    ipcRenderer.on('response-outputjson', (event, functionNamesJson) => {
        functionNamesJson.forEach(element => {
        let lastCluster = null;
        let currentScopeBlock = null;
        let scopeName = null;
        var index = null;
            if (element.scope !== 'global' && element.translate) {
                const classBlock = ws.newBlock('python_class');
                classBlock.setFieldValue(element.base_classes, 'CLASS_NAME');
                classBlock.initSvg();
                classBlock.render();
                currentScopeBlock = classBlock
                scopeName = element.scope;
                index = element.row;

            }

            element.functions.forEach(func => {
                if (!func.translate) return;

                if(index==null) index = func.row;
                if(scopeName===null){
                    scopeName=func.functionName;
                }
                const functionBlock = ws.newBlock('python_function');
                functionBlock.setFieldValue(func.functionName + func.parameters, 'CLASS_NAME');
                functionBlock.initSvg();
                functionBlock.render();
                if(lastCluster!==functionBlock && lastCluster!==null){
                    const lastBlockConnection = lastCluster.nextConnection;
                    const currentBlockConnection = functionBlock.previousConnection;

                    if (lastBlockConnection && currentBlockConnection && lastCluster) {
                        lastBlockConnection.connect(currentBlockConnection);
                    }

                }
                const comment = ws.newBlock('comment');
                comment.initSvg();
                comment.render();

                appendBlockToFunction(functionBlock, comment);
                lastBlock = comment;
                if (currentScopeBlock) appendBlockToClass(currentScopeBlock, functionBlock);


                let currentBlock = null;
                currentBlock = comment
                func.functionCalls.forEach(call => {
                    const blockInfo = getBlockInformation(call);
                    if (currentBlock ) {
                        appendBlockToWorkspace(blockInfo);
                    }
                    ws.resizeContents();
                });

                lastCluster  = functionBlock;
                if(currentScopeBlock!=null){
                    lastCluster = currentScopeBlock;

                }
            });


                let select = document.getElementById('ViewList');


                const workspaceState = Blockly.serialization.workspaces.save(ws);

                WorkspaceStates[select.length-1] = JSON.stringify(workspaceState);
                var value = 0;
                if(select.length!==0){
                    console.log("Inserting at " + select.length)
                    value=select.length-1;
                }




                select.add(new Option(scopeName,value));

                currentWS  =value+1;
                ws.clear();
                ipcRenderer.send('create-new-view', scopeName,index);

                scopeName=null;
                currentScopeBlock=null;

        });
        document.getElementById('CreateHandle').style.display="flex"
        document.getElementById('swap').style.display="flex"
});

document.getElementById('CodeToBlock').addEventListener('click', () => {
ipcRenderer.send('read-outputjson');
    const filePath = path.join('./projectsrc/projectskeleton2.py'); //Reads the generated file

    fs.readFile(filePath, 'utf-8', (err, data) => {
        console.log("reading...")
        if (err) {
            console.error('Failed to load file:', err);
            return;
        }
        editor.setValue(data, -1);
        document.getElementById('CodeToBlock').style.display="none";
    });
});
});
