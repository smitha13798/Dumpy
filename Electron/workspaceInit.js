console.log('Attempting to load toolbox module...');
const toolboxPath = './toolbox.js';

document.addEventListener('DOMContentLoaded', async function () {
    let toolbox;
    try {
        const module = await import(toolboxPath);
        toolbox = module.toolbox;
        console.log('Toolbox loaded:', toolbox);
    } catch (error) {
        console.error('Error loading toolbox:', error);
    }

    const Blockly = require('blockly');
    const blocklyDiv = document.getElementById('blocklyDiv');
    if (blocklyDiv) {
        console.log('blocklyDiv found:', blocklyDiv);
        try {
            const workspace = Blockly.inject(blocklyDiv, {toolbox});
            console.log('Blockly has been initialized', workspace);
        } catch (error) {
            console.error('Failed to initialize Blockly:', error);
        }
    } else {
        console.log('Failed to find blocklyDiv');
    }
});
