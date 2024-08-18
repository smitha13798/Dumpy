
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
var currentWS = 0;
const { ipcRenderer } = require('electron');

function AddSubOption(optionTag, options) {
    var OptionTag = document.getElementById(optionTag);
    OptionTag.style.display = 'inline';
    for (let i = 0; i < options.length; i++) {
        OptionTag.add(new Option(options[i]));
    }
}

function getMaxValue() {
    // Get the select element
    let selectElement = document.getElementById('ViewList');

    // Initialize the maximum value
    let maxValue = -Infinity;

    // Iterate through the options to find the maximum value
    for (let i = 0; i < selectElement.options.length; i++) {
        let optionValue = parseFloat(selectElement.options[i].value);
        if (optionValue > maxValue) {
            maxValue = optionValue;
        }
    }

    // Display the maximum value
    return maxValue;
}



document.addEventListener('DOMContentLoaded', function () {
    const createNewViewButton = document.getElementById('createNewView');
    var buttonState = 0;
    var viewName = "";



    createNewViewButton.addEventListener('click', () => {

        if(buttonState===0){
            viewName = document.getElementById('new-view-name').value.trim();
            document.getElementById('new-view-name').value = "";
            document.getElementById('new-view-name').placeholder = 'Index...';

            buttonState=1;
            return;
        }
        let newOption = document.createElement('option');
        const index = document.getElementById('new-view-name').value.trim();

        // Step 2: Set the text and value of the new <option> element
        newOption.text = viewName+index+"";
        newOption.value = getMaxValue()+1+"";

        // Step 3: Append the new <option> element to the <select> element

        document.getElementById('ViewList').add(newOption);
        ipcRenderer.send('create-new-view',viewName,index);
        buttonState=0;
    });


    const swapButton = document.getElementById('swap');
    swapButton.addEventListener('click', function () {
        const code = document.getElementById('textarea').textContent;
        ipcRenderer.send('save-code-to-file', code);
        ws.clear();
        ipcRenderer.send('change-view');
    });
});





document.addEventListener('DOMContentLoaded', async function () {
    let toolbox;
    let blocks;
    let load;
    let save;
    let forBlock;
    const codeToBlock = document.getElementById('CodeToBlock');
    codeToBlock.addEventListener('click', function() {
        console.log("Going to server now")
        ipcRenderer.send('read-outputjson');
    });
    ipcRenderer.on('response-outputjson', (event, functionNamesJson) => {
        for(const element of functionNamesJson){

            console.log("Scope is " + element.scope)
            for(const functions of element.functions){
                if(!functions.translate){
                    continue;
                }
                console.log("def "+ functions.functionName )
                for(const functionCalls of functions.functionCalls){
                    console.log(functionCalls)
                }
            }

        }
    });

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
    } catch (error) {
        console.error('Error loading blocks:', error);
    }
    try {
        const module = await import(serializerPath);
        load = module.load;
        save = module.save;
    } catch (error) {
        console.error('Error loading serializer:', error);
    }
    try {
        const module = await import(javascriptPath);
        forBlock = module.forBlock;
    } catch (error) {
        console.error('Error loading JavaScript generator:', error);
    }

    const Blockly = require('blockly');
    const pythonGenerator = require('blockly/python');
    const javaScriptGenerator = require('blockly/javascript');
    Blockly.common.defineBlocks(blocks);
    Object.assign(javaScriptGenerator.javascriptGenerator.forBlock, forBlock);









    const blocklyDiv = document.getElementById('blocklyDiv');
    if (blocklyDiv) {
        console.log('blocklyDiv found:', blocklyDiv);
        try {
            ws = Blockly.inject(blocklyDiv, { toolbox });
            console.log('Blockly has been initialized', ws);
        } catch (error) {
            console.error('Failed to initialize Blockly:', error);
        }
    } else {
        console.log('Failed to find blocklyDiv');
    }

    const runCode = () => {
        const code = pythonGenerator.pythonGenerator.workspaceToCode(ws);
        console.log("Code comes here: " + code);
    };

    load(ws); // Load the initial state for the workspace
    runCode();

    ws.addChangeListener((e) => {
        if (e.isUiEvent) return;
        save(ws);
    });

    ws.addChangeListener((e) => {
        if (e.isUiEvent || e.type == Blockly.Events.FINISHED_LOADING || ws.isDragging()) {
            console.log("Dragging...");
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
        if (ws.isDragging()) return;
        if (!supportedEvents.has(event.type)) return;

        const code = pythonGenerator.pythonGenerator.workspaceToCode(ws);
        console.log("changed");
        document.getElementById('textarea').textContent = code;
        //ipcRenderer.send('save-code-to-file', code);
    }
    const saveButton = document.getElementById('saveButton');
    saveButton.onclick = function(){
        ipcRenderer.send('save-code-to-file', pythonGenerator.pythonGenerator.workspaceToCode(ws));
    }
    ws.addChangeListener(updateCode);

    const viewList = document.getElementById('ViewList');

    viewList.onchange = function () {
        const selectedIndex = viewList.value;

        // Save the current workspace state before switching
        const currentXml = Blockly.Xml.workspaceToDom(ws);
        WorkspaceStates[currentWS] = Blockly.Xml.domToText(currentXml);
        currentWS = selectedIndex;
        // Clear the current workspace
        const code = pythonGenerator.pythonGenerator.workspaceToCode(ws);

        //ipcRenderer.send('save-code-to-file', code);
        ipcRenderer.send('change-view-option', selectedIndex);

        ws.clear();

        // Load the selected workspace state
        if (WorkspaceStates[selectedIndex]) {
            const newXml = Blockly.utils.xml.textToDom(WorkspaceStates[selectedIndex]);
            Blockly.Xml.domToWorkspace(newXml, ws);
        }

        runCode();
        if (selectedIndex === '1') {
            console.log("adding options...");
            AddSubOption('subViewList', [7, 22]);
            return;
        }
        document.getElementById('subViewList').style.display = "none";
    };
});
