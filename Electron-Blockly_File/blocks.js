/**
 * @license
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

const Blockly = require('blockly/core');
// Create a custom block called 'add_text' that adds
// text to the output div on the sample app.
// This is just an example and you should replace this with your
// own custom blocks.
const python_class = {
  "type": "python_class",
  "message0": "class %1 :%2 %3",
  "args0": [
      {
          "type": "field_input",
          "name": "CLASS_NAME",
          "text": "MyClass"
      },
      {
          "type": "input_dummy"
      },
      {
          "type": "input_statement",
          "name": "METHODS"
      }
  ],
  "colour": 230,
  "tooltip": "Define a Python class with methods and attributes.",
  "helpUrl": ""
};


const addText = {
  'type': 'add_text',
  'message0': 'Add text %1 with color %2',
  'args0': [
    {
      'type': 'input_value',
      'name': 'TEXT',
      'check': 'String',
    },
    {
      'type': 'input_value',
      'name': 'COLOR',
      'check': 'Colour',
    },
  ],
  'previousStatement': null,
  'nextStatement': null,
  'colour': 160,
  'tooltip': '',
  'helpUrl': '',
};
const DataWrapperG = {
  'type': 'DataWrapper',
  'message0': 'Add Data source %1 ', //%1 is referencing to input value, must be named!
  'args0': [
    {
      'type': 'input_value',
      'name': 'TEXT',
      'check': '',
    }
  ],
  'previousStatement': null,
  'nextStatement': null,
  'colour': 160,
  'tooltip': '',
  'helpUrl': '',
};

const AddVectors = {
  "type": "Add_vectors",
  "message0": "Add vector%1 %2",
  "args0": [
    {
      "type": "input_value",
      "name": "Array1",
      "check": "Array"
    },
    {
      "type": "input_value",
      "name": "Array2",
      "check": "Array",
      "align": "RIGHT"
    }
  ],
  "colour": 230,
  "tooltip": "",
  "helpUrl": ""
};
const generateRandome = {
  "type": "generate_randomePRNG",
  "message0": "Generate Randome PRNGKey %1",
  "args0": [
    {
      "type": "input_value",
      "name": "seed",
      "check": ""
    },
  ],
  "colour": 230,
  "tooltip": "",
  "helpUrl": ""
};
const flattenLayer = {
  'type': 'flatten_layer',
  'message0': 'Flatten tensor',
  'previousStatement': null,
  'nextStatement': null,
  'colour': 210,
  'tooltip': 'Flattens the input tensor into a single continuous vector.',
  'helpUrl': ''
};
const denseLayer = {
  'type': 'dense_layer',
  'message0': 'Dense layer with %1 units',
  'args0': [
    {
      'type': 'field_number',
      'name': 'UNITS',
      'value': 10,  // Default number of units
      'min': 1
    }
  ],
  'previousStatement': null,
  'nextStatement': null,
  'colour': 230,
  'tooltip': 'Adds a dense layer with specified number of units.',
  'helpUrl': ''
};
const maxPoolLayer = {
  'type': 'max_pool_layer',
  'message0': 'MaxPool with window shape: %1 %2 strides: %3 %4',
  'args0': [
    {
      'type': 'field_number',
      'name': 'WINDOW_SHAPE_X',
      'value': 2,  // Default window shape X
      'min': 1
    },
    {
      'type': 'field_number',
      'name': 'WINDOW_SHAPE_Y',
      'value': 2,  // Default window shape Y
      'min': 1
    },
    {
      'type': 'field_number',
      'name': 'STRIDE_X',
      'value': 2,  // Default stride X
      'min': 1
    },
    {
      'type': 'field_number',
      'name': 'STRIDE_Y',
      'value': 2,  // Default stride Y
      'min': 1
    }
  ],
  'previousStatement': null,
  'nextStatement': null,
  'colour': 120,
  'tooltip': 'Applies a MaxPooling operation with specified window shape and strides.',
  'helpUrl': ''
};



const relu = {
  "type": "relu",
  "message0": "Call relu  %1",
  "args0": [
    {
      "type": "input_value",
      "name": "model",
      "check": ""
    },

  ],
  "output": "Linear Layer",
  "colour": 230,
  "tooltip": "Give linear layer",
  "helpUrl": ""
}

const conv = {
  "type": "Conv",
  "message0": "Create conv layer %1 %2",
  "args0": [
    {
      "type": "input_value",
      "name": "feature",
      "check": ""
    },
    {
      "type": "input_value",
      "name": "kernel",
      "check": ""
    }

  ],
  "output": "ConvLayer",
  "colour": 230,
  "tooltip": "Define a convolutional layer",
  "helpUrl": ""
}



// Create the block definitions for the JSON-only blocks.
// This does not register their definitions with Blockly.
// This file has no side effects!
export const blocks = Blockly.common.createBlockDefinitionsFromJsonArray(
  [python_class,DataWrapperG, addText, AddVectors, generateRandome, flattenLayer, denseLayer, maxPoolLayer,relu,conv]
);


// Create the block definitions for the JSON-only blocks.
// This does not register their definitions with Blockly.
// This file has no side effects!
