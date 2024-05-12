/**
 * @license
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as Blockly from 'blockly';
import {blocks} from './blocks/text';

import {forBlock} from './generators/javascript';

import {pythonGenerator} from "blockly/python";
import {javascriptGenerator} from "blockly/javascript";
import {save, load} from './serialization';
import {toolbox} from './toolbox';
import './index.css';

// Register the blocks and generator with Blockly
Blockly.common.defineBlocks(blocks);
Object.assign(javascriptGenerator.forBlock, forBlock);

// Set up UI elements and inject Blockly
const codeDiv = document.getElementById('generatedCode').firstChild;
const outputDiv = document.getElementById('output');
const blocklyDiv = document.getElementById('blocklyDiv');
const ws = Blockly.inject(blocklyDiv, {toolbox});

// This function resets the code and output divs, shows the
// generated code from the workspace, and evals the code.
// In a real application, you probably shouldn't use `eval`.


  const runCode = () => {
    const code = pythonGenerator.workspaceToCode(ws)
    codeDiv.innerText = code;

    outputDiv.innerHTML = '';
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

  const code = pythonGenerator.workspaceToCode(ws);
  console.log("changed")
  document.getElementById('textarea').textContent = code;
}

ws.addChangeListener(updateCode);
