/**
 * @license
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */


const {Order,pythonGenerator} = require('blockly/python')
// Export all the code generators for our custom blocks,
// but don't register them with Blockly yet.
// This file has no side effects!
export const forBlock = Object.create(null);
pythonGenerator.forBlock['python_class'] = function(block) {
    const className = block.getFieldValue('CLASS_NAME');
    const methods = pythonGenerator.statementToCode(block, 'METHODS');
    return `class ${className}:\n${methods}`;
};
pythonGenerator.forBlock['relu'] = function(block, generator) {
    var model = generator.valueToCode(block, 'model', Order.NONE);
    // TODO: Assemble python into code variable.
    // TODO: Change ORDER_NONE to the correct strength.
    return 'nn.relu('+model+')';
};

pythonGenerator.forBlock['Conv'] = function(block, generator) {
    var value_feature = generator.valueToCode(block, 'feature', Order.NONE);
    var value_kernel_size = generator.valueToCode(block, 'kernel', Order.NONE);
    // TODO: Assemble python into code variable.
    // TODO: Change ORDER_NONE to the correct strength.
    return 'nn.Conv('+value_feature+','+value_kernel_size+')';
};
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




pythonGenerator.forBlock['generate_randomePRNG'] = function (block, generator) {

    const seed = generator.valueToCode(block, 'seed', Order.NONE) || "''";


    const addText = generator.provideFunction_(
        'generate_randomePRNG',
        `function ${generator.FUNCTION_NAME_PLACEHOLDER_}(seed) {
       key = jax.random.PRNGKey(seed)
       return key
}`
    );
    return `${addText}(${seed});\n`;

    // Generate the function call for this block.

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
pythonGenerator.forBlock['flatten_layer'] = function(block) {
    return 'x = x.reshape((x.shape[0], -1))\n';
  };
  
pythonGenerator.forBlock['dense_layer'] = function(block) {
    var units = block.getFieldValue('UNITS');
    return `x = nn.Dense(features=${units})(x)\n`;
  };
pythonGenerator.forBlock['max_pool_layer'] = function(block) {
    var windowShapeX = block.getFieldValue('WINDOW_SHAPE_X');
    var windowShapeY = block.getFieldValue('WINDOW_SHAPE_Y');
    var strideX = block.getFieldValue('STRIDE_X');
    var strideY = block.getFieldValue('STRIDE_Y');
    return `x = nn.MaxPool(window_shape=(${windowShapeX}, ${windowShapeY}), strides=(${strideX}, ${strideY}))(x)\n`;
  };





    













