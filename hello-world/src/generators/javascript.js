/**
 * @license
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */


//import {Order} from 'blockly/python';
import {Order, pythonGenerator} from 'blockly/python'
// Export all the code generators for our custom blocks,
// but don't register them with Blockly yet.
// This file has no side effects!
export const forBlock = Object.create(null);
pythonGenerator.forBlock['DataWrapper'] = function (block, generator) {
  const text = generator.valueToCode(block, 'TEXT', Order.NONE) || "''";


  const addText = generator.provideFunction_(
      'addText',
      `function ${generator.FUNCTION_NAME_PLACEHOLDER_}(text) {

  // Add text to the output area.
  const outputDiv = document.getElementById('output');
  const textEl = document.createElement('p');
  textEl.innerText = text+ "HEEEEEY";
  outputDiv.appendChild(textEl);
}`
  );

  // Generate the function call for this block.
  const code = `${addText}(${text});\n`;

  return code;
};

pythonGenerator.forBlock['Add_vectors'] = function (block, generator) {

  const vector1 = generator.valueToCode(block, 'Array1', Order.NONE) || "''";
  const vector2 = generator.valueToCode(block, 'Array2', Order.NONE) || "''";


  const addText = generator.provideFunction_(
      'AddVectors',
      `function ${generator.FUNCTION_NAME_PLACEHOLDER_}(vector1,vector2) {
       jnp.add(vector1, vector2)
}`
  );
  return `${addText}(${vector1},${vector2});\n`;

  // Generate the function call for this block.

};

pythonGenerator.forBlock['add_text'] = function (block, generator) {
  const text = generator.valueToCode(block, 'TEXT', Order.NONE) || "''";
  const color =
    generator.valueToCode(block, 'COLOR', Order.ATOMIC) || "'#ffffff'";

  const addText = generator.provideFunction_(
      'addText',
      `function ${generator.FUNCTION_NAME_PLACEHOLDER_}(text, color) {

  // Add text to the output area.
  const outputDiv = document.getElementById('output');
  const textEl = document.createElement('p');
  textEl.innerText = text;
  textEl.style.color = color;
  outputDiv.appendChild(textEl);
}`
  );


  // Generate the function call for this block.
  const code = `${addText}(${text}, ${color});\n`;
  return code;

};













