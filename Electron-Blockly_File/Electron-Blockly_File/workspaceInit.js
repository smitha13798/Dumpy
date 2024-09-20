

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
let destinationPath = null;
// Utility function to add sub-options to a dropdown
const editor = ace.edit("codeeditor");
const editorCurrent = ace.edit('currentEditor');
editorCurrent.setTheme("ace/theme/monokai");
editorCurrent.session.setMode("ace/mode/python");
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
let whileBlock = null;
let lastBlockIndex = null;
// Function to append a block to the workspace and manage connections

function connectBlocksVertical(lastBlock,block){
    if (lastBlock) {
        // Ensure that connections are valid before attempting to connect

        const lastBlockConnection = lastBlock.nextConnection;
        const currentBlockConnection = block.previousConnection;
        // Debugging: Check if connections exist


        if (lastBlockConnection && currentBlockConnection) {
            lastBlockConnection.connect(currentBlockConnection);
        } else {
            console.error('Connection points are not available for lastBlock.');
        }
    }
}
function isNumericString(str) {
    // Try to convert the string to a number
    return !isNaN(parseFloat(str)) && isFinite(str);
}
function connectBlocksHorizontaly(leftBlock,rightBlock){
    const valueInput = leftBlock.getInput('VALUE');
    const valueInputConnection = valueInput ? valueInput.connection : null;
    const functionOutputConnection = rightBlock.outputConnection;

    // Debugging: Check if connections exist


    if (valueInputConnection && functionOutputConnection) {
        // Connect the variable block to the function block
        valueInputConnection.connect(functionOutputConnection);

    } else {
        console.error('One or both connections are not available for variable block.');
    }
}

function extractXFromExpression(expression) {
    // Convert expression to a string if it's not already
    expression = expression.toString();

    // Use a regular expression to match the pattern (x) at the end of the expression
    const match = expression.match(/\)\(([^)]+)\)$/);

    if (match) {
        const content = match[1]; // Extract the content inside the parentheses

        // Remove the last pair of parentheses and its content from the original string
        // Additionally, remove the first '(' in the updated expression
        let updatedExpression = expression.replace(/\)\([^)]+\)$/, '');

        // Remove the first '(' if it exists in the updated expression
        updatedExpression = updatedExpression.replace(/^\(/, '');

        return {
            content: content,
            updatedExpression: updatedExpression
        };
    }

    // If no match is found, return the original expression and null content
    return {
        content: null,
        updatedExpression: expression
    };
}
function removeFirstAndLastParentheses(str) {
    console.log("String is " + str)
    // Remove the first opening parenthesis
    let result = str.replace(/^\(/, '');
    // Remove the last closing parenthesis
    result = result.replace(/\)$/, '');
    console.log("is now .. " + result)
    return result;
}
function appendBlockToWorkspace(blockInfo) {

    // Create and configure the new block
    let block= null;
    if(blockInfo.name==="range"){
        return;
    }
    if(blockInfo.name ==="Variable"){
        var MultiVariable = [];
        MultiVariable = blockInfo.assigned.split(',');
        if(MultiVariable.length>0){
            for(const element of MultiVariable) {
                block = ws.newBlock('set_var');
                block.setFieldValue(element,"SET_VARIABLE")

                block.initSvg();
                block.render();
                if(blockInfo.parameters!=="" && blockInfo.name!=="cheatblock"){
                    let assignmentValue =null;
                    if(isNumericString(blockInfo.parameters)){

                        assignmentValue = ws.newBlock('math_number');
                        assignmentValue.setFieldValue(blockInfo.parameters,'NUM')
                        assignmentValue.render()
                        assignmentValue.initSvg();
                        connectBlocksHorizontaly(block,assignmentValue)
                    }
                    else{
                        assignmentValue = ws.newBlock('string');
                        assignmentValue.setFieldValue(blockInfo.parameters,'VAR')
                        assignmentValue.render()
                        assignmentValue.initSvg();
                        connectBlocksHorizontaly(block,assignmentValue)
                    }

                }

                connectBlocksVertical(lastBlock,block)

                // Update the lastBlock reference
                lastBlock = block;


                // Force a workspace update
                ws.resizeContents();
            }
        }


        return;
    }

    else{
        if(blockInfo.name==="python_loop"){
            block = ws.newBlock('python_loop')
            block.setFieldValue(blockInfo.assigned,"element");
            block.setFieldValue(blockInfo.parameters,"FIELD")
            block.render();
            block.initSvg()
            whileBlock = block;


        }

        else{ //Normal void block
            block = ws.newBlock(blockInfo.name);
            var str =  blockInfo.parameters.toString();
            let inputValue = blockInfo.parameters.toString();
            const result = extractXFromExpression(inputValue.split(','));
            const specialInput =  result.content

            if (inputValue === "()") inputValue = "";

            // Configure the new block with its parameters
            var Inputs = [];
            block.inputList.forEach(input => {
                if (input.fieldRow) {
                    input.fieldRow.forEach(field => {
                        if (field.name !== undefined) {
                            Inputs.push(field.name);
                            /*block.setFieldValue(inputValue, field.name);
                            if (field.name==="VAR"){

                                block.setFieldValue(specialInput, field.name);

                            }
                            inputValue=""*/
                        }
                    });
                }

            });
            if(Inputs.length>1){ //HAS SPECIAL INPUT AT THE END

                const result = extractXFromExpression(inputValue.split(','))
                block.setFieldValue(result.updatedExpression,Inputs[0])
                block.setFieldValue(result.content,Inputs[1])
            }

                if(Inputs[0]===undefined){
                    let variableBlock = null;
                    console.log("CALLED cuz undefined")
                    return;
                }
                else {


                    const result = extractXFromExpression(inputValue.split(','))

                    if(result.content!==null) {
                        block.setFieldValue(result.updatedExpression, Inputs[0])
                    }
                    else{
                        block.setFieldValue(inputValue,Inputs[0])
                        if(blockInfo.name!=="cheatblock"){
                            str = removeFirstAndLastParentheses(str)
                            block.setFieldValue(str,Inputs[0])
                        }

                    }



                }


        }




    }

    let variableBlock = null;

    // If there's an assignment, create a variable block




        if (blockInfo.assigned && blockInfo.name !== "Variable" && blockInfo.name != "python_loop" && blockInfo.name!="cheatblock") {

            variableBlock = ws.newBlock('set_var');
            variableBlock.setFieldValue(blockInfo.assigned, 'SET_VARIABLE');
            variableBlock.initSvg();
            variableBlock.render();

        }



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


        if (valueInputConnection && functionOutputConnection) {
            // Connect the variable block to the function block
            valueInputConnection.connect(functionOutputConnection);
            block = variableBlock;
        } else {
            console.error('One or both connections are not available for variable block.');
        }

    }

    // Connect to the previous lastBlock if applicable
    connectBlocksVertical(lastBlock,block)

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
    let editor = ace.edit("codeeditor");
    editor.setTheme("ace/theme/monokai");
    editor.session.setMode("ace/mode/python");
    let editorCurrent = ace.edit("currentEditor");
    editorCurrent.setTheme("ace/theme/monokai");
    editorCurrent.session.setMode("ace/mode/python")


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
            WorkspaceStates[select.length] = JSON.stringify(workspaceState);
            let value = 0;
            if(select.length!==0){
                value=select.length-1;
            }

            select.add(new Option(`${viewName}${index}`,value))
            currentWS  =value;
            ws.clear();
            select.value  = currentWS;
            ipcRenderer.send('create-new-view-with-index', `${viewName}`,index);
            buttonState = 0;
            document.getElementById('new-view-name').value = "";
            document.getElementById('new-view-name').placeholder = 'Enter view name';


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
    const loadButton = document.getElementById('loadFileButton');
    loadButton.onclick = () => {
        ipcRenderer.send('start-script');

        ipcRenderer.on('load-blocks', (event, filePath) => {
            ipcRenderer.send('read-outputjson');
            if(filePath!==""){
                destinationPath = filePath;

            }


            const skeletonPath = path.join('./projectsrc/projectskeleton2.py'); // Reads the generated file

            fs.readFile(skeletonPath, 'utf-8', (err, data) => {

                if (err) {
                    console.error('Failed to load file:', err);
                    return;
                }
                editor.setValue(data, -1);
                document.getElementById('CodeToBlock').style.display = "none";
            });
            document.getElementById('loadFileButton').style.display="none";
            document.getElementById('saveButton').style.display="flex";

        });

        ipcRenderer.on('script-error', (event, errorMessage) => {
            console.error('Error in script execution:', errorMessage);
            // Display the error message to the user or log it
            alert(`Script error: ${errorMessage}`);
        });
    };

    const saveButton = document.getElementById('saveButton');
    saveButton.onclick = () => {


        // Save current workspace state before switching
        const workspaceState = Blockly.serialization.workspaces.save(ws);

        WorkspaceStates[currentWS] = JSON.stringify(workspaceState);
        var codeArray = [];
        for(var i=0;i<WorkspaceStates.length;i++){

            // Save current workspace state before switching


            const workspaceStateString = WorkspaceStates[i];

// Parse the state string back to a JSON object
            const offScreenWs = new Blockly.Workspace();

// Parse the state string back to a JSON object
            const workspaceState = JSON.parse(workspaceStateString);
// Load the workspace state into the off-screen workspace
            Blockly.serialization.workspaces.load(workspaceState, offScreenWs);

// Generate the code from the off-screen workspace
            codeArray[i] = pythonGenerator.pythonGenerator.workspaceToCode(offScreenWs);

// Dispose of the off-screen workspace to free up resources
            offScreenWs.dispose();



        }
        document.getElementById('CreateHandle').style.display="none"
        document.getElementById('saveButton').style.display="none"
        document.getElementById('searchBox').style.display="none";
        document.getElementById('textarea').style.display="none";
        document.getElementById('progress').style.display="flex";
        document.getElementById('blocklyDiv').style.display = "none";
        document.getElementById('codeeditor').style.display = "none";
        ipcRenderer.send('save-block-to-file',codeArray)



        setTimeout(() => {
            document.getElementById('')

            document.getElementById('currentEditor').style.display = "flex";
            document.getElementById('codeeditor').style.display = "flex";
            document.getElementById('progress').style.display="absolute";
            document.getElementById('progress').style.display="none";

            const filePathOriginal = path.join('./projectsrc/projectsrc.py'); //Reads the generated file

            fs.readFile(filePathOriginal, 'utf-8', (err, data) => {

                if (err) {
                    console.error('Failed to load file:', err);
                    return;
                }
                editorCurrent.setValue(data, -1);



            });
            ws.clear()
            const filePathNew = path.join('./projectsrc/projectskeleton2.py'); //Reads the generated file

            fs.readFile(filePathNew, 'utf-8', (err, data) => {

                if (err) {
                    console.error('Failed to load file:', err);
                    return;
                }
                editor.setValue(data, -1);



            });
            document.getElementById('discard').style.display="flex"
            document.getElementById('replace').style.display="flex"
        }, 2000); // 2000 milliseconds = 2 seconds


    };
    const submit = document.getElementById('replace');
    submit.onclick = function (){
        var updatedCode = editor.getValue()

        ipcRenderer.send('submit-changes',updatedCode,destinationPath);

    }
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
        let index = null;

            if (element.scope !== 'global' && element.translate) {
                const classBlock = ws.newBlock('python_class');
                classBlock.setFieldValue(element.scope+element.base_classes, 'CLASS_NAME');
                classBlock.initSvg();
                classBlock.render();
                currentScopeBlock = classBlock
                scopeName = element.scope;
                index = element.row;
                const comment = ws.newBlock('comment');
                comment.initSvg();
                comment.render();
                appendBlockToFunction(classBlock, comment);
                lastBlock = comment;

                    let declarations = element.class_declaration;

                    for(const element of declarations) {
                    if(element.parameters==="@nn.compact") {

                        const declartaion = ws.newBlock("nn_compact");
                        declartaion.render();
                        declartaion.initSvg();
                        connectBlocksVertical(lastBlock, declartaion)
                    }
                        else{
                            if(element.parameters==="@nn.nowrap"){
                                const declartaion = ws.newBlock("nn_nowrap");
                                declartaion.render();
                                declartaion.initSvg();
                                connectBlocksVertical(lastBlock, declartaion)
                            }
                            else{
                                const declartaion = ws.newBlock('Declaration');
                                declartaion.setFieldValue(element.parameters, 'SET_VARIABLE');
                                declartaion.render();
                                declartaion.initSvg();
                                connectBlocksVertical(lastBlock,declartaion)
                            }

                        }



                    }

                if(element.attributes.length>0){
                    let memberArr = element.attributes;

                    for(const element of memberArr) {
                        const member = ws.newBlock('member');
                        member.setFieldValue(element.assigned, 'SET_VARIABLE');
                        member.render();
                        member.initSvg();
                        const type = ws.newBlock('string');
                        type.setFieldValue(element.parameters,'VAR');
                        type.render();
                        type.render();
                        connectBlocksHorizontaly(member,type)
                        connectBlocksVertical(lastBlock,member)

                    }

                }
            }

            element.functions.forEach(func => {
                const returnValue = func.returns;
                if (!func.translate) return;


                if(index==null) index = func.row;
                if(scopeName===null){
                     lastCluster = null;
                     currentScopeBlock = null;
                     scopeName = null;

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
                        if(call.index!==lastBlockIndex){
                            appendBlockToWorkspace(blockInfo);
                            lastBlockIndex = call.index;
                        }


                    }
                    if(call.function==="python_loop"){


                        if(whileBlock!==null) {
                            const comment = ws.newBlock('comment');
                            comment.render();
                            comment.initSvg();
                            appendBlockToFunction(whileBlock,comment)
                            var temp = lastBlock;
                            lastBlock = comment;
                            const functionCallsLoop = call.functionCallsLoop;
                            let lastwhileblockIndex = null;
                            for (const element of functionCallsLoop) {
                                if(lastwhileblockIndex!==element.index){
                                    const blockInfo = getBlockInformation(element)
                                    appendBlockToWorkspace(blockInfo)
                                    lastwhileblockIndex = element.index
                                }



                            }
                        }

                        whileBlock=null;
                        lastBlock = temp;
                    }



                    ws.resizeContents();
                });

                if(returnValue){
                    const returnBlock = ws.newBlock('python_return');

                    returnBlock.initSvg();
                    returnBlock.render();

                    returnBlock.setFieldValue(returnValue,'RETURN_VALUE')
                    const lastBlockConnection = lastBlock.nextConnection;
                    const currentBlockConnection = returnBlock.previousConnection;


                    if (lastBlockConnection && currentBlockConnection) {
                        lastBlockConnection.connect(currentBlockConnection);
                    } else {
                        console.error('Connection points are not available for lastBlock.');
                    }
                }




                lastCluster  = functionBlock;
                if(currentScopeBlock!=null){
                    lastCluster = currentScopeBlock;
                }

                if (currentScopeBlock === null) {


                    let select = document.getElementById('ViewList');


                    const workspaceState = Blockly.serialization.workspaces.save(ws);

                    WorkspaceStates[select.length - 1] = JSON.stringify(workspaceState);
                    let value = 0;
                    if (select.length !== 0) {
                        value = select.length - 1;
                    }


                    select.add(new Option(scopeName, value + ""));

                    currentWS = value + 1;
                    ws.clear();
                    ipcRenderer.send('create-new-view', scopeName, index);
                    scopeName=null;
                    currentScopeBlock=null;
                }


            if(!currentScopeBlock){
                index=null;
            }

            });

                if(currentScopeBlock!==null) {


                    let select = document.getElementById('ViewList');


                    const workspaceState = Blockly.serialization.workspaces.save(ws);

                    WorkspaceStates[select.length - 1] = JSON.stringify(workspaceState);
                    let value = 0;
                    if (select.length !== 0) {
                        value = select.length - 1;
                    }


                    select.add(new Option(scopeName, value + ""));

                    currentWS = value + 1;
                    ws.clear();
                    ipcRenderer.send('create-new-view', scopeName, index);
                    scopeName=null;
                    currentScopeBlock=null;
                }


        });
            document.getElementById('CreateHandle').style.display="flex"
        document.getElementById('swap').style.display="flex"
});

document.getElementById('CodeToBlock').addEventListener('click', () => {
ipcRenderer.send('read-outputjson');
    const filePath = path.join('./projectsrc/projectskeleton2.py'); //Reads the generated file

    fs.readFile(filePath, 'utf-8', (err, data) => {
        if (err) {
            console.error('Failed to load file:', err);
            return;
        }
        editor.setValue(data, -1);


        document.getElementById('CodeToBlock').style.display="none";
    });
});






});
