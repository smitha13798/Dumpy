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

const python_function = {
  "type": "python_function",
  "message0": "function %1 :%2 %3",
  "args0": [
    {
      "type": "field_input",
      "name": "CLASS_NAME",
      "text": "myFunction"
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
  'previousStatement': null,
  'nextStatement': null,
  "tooltip": "Define a Python class with methods and attributes.",
  "helpUrl": ""
};

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
    'previousStatement': null,
  'nextStatement': null,
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
    'previousStatement': null,
  'nextStatement': null,
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
  "type": "relu_layer",
  "message0": "ReLU activation",
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Apply ReLU activation",
  "helpUrl": ""
};

const conv = {
  "type": "conv_layer",
  "message0": "Conv layer with features %1 kernel size %2 %3",
  "args0": [
    {
      "type": "field_number",
      "name": "FEATURES",
      "value": 64, // default number of features
      "min": 1
    },
    {
      "type": "field_number",
      "name": "KERNEL_SIZE_X",
      "value": 3, // default kernel size X
      "min": 1
    },
    {
      "type": "field_number",
      "name": "KERNEL_SIZE_Y",
      "value": 3, // default kernel size Y
      "min": 1
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Apply a convolutional layer",
  "helpUrl": ""
};
const averagePool = {
  "type": "average_pool_layer",
  "message0": "Average Pool with pool size: %1 %2 strides: %3 %4",
  "args0": [
    {
      "type": "field_number",
      "name": "POOL_SIZE_X",
      "value": 2,  // Default pool size X
      "min": 1
    },
    {
      "type": "field_number",
      "name": "POOL_SIZE_Y",
      "value": 2,  // Default pool size Y
      "min": 1
    },
    {
      "type": "field_number",
      "name": "STRIDE_X",
      "value": 2,  // Default stride X
      "min": 1
    },
    {
      "type": "field_number",
      "name": "STRIDE_Y",
      "value": 2,  // Default stride Y
      "min": 1
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Apply an average pooling layer",
  "helpUrl": ""
};
const dropout = {
  "type": "dropout_layer",
  "message0": "Dropout with rate %1",
  "args0": [
    {
      "type": "field_number",
      "name": "RATE",
      "value": 0.5,  // Default dropout rate
      "min": 0,
      "max": 1
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Apply dropout for regularization",
  "helpUrl": ""
};

const batchNorm = {
  "type": "batch_norm_layer",
  "message0": "Batch Normalization",
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Apply batch normalization",
  "helpUrl": ""
};

const self = {
  "type": "self",
  "message0": "Init self  %1",
  'previousStatement': null,
  'nextStatement': null,
  "args0": [
    {
      "type": "input_value",
      "name": "func",
      "check": ""
    },

  ],
  "output": "Linear Layer",
  "colour": 230,
  "tooltip": "Give linear layer",
  "helpUrl": ""
}
const tanh = {
  "type": "tanh_layer",
  "message0": "Tanh activation",
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Apply Tanh activation",
  "helpUrl": ""
};
const sigmoid = {
  "type": "sigmoid_layer",
  "message0": "Sigmoid activation",
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Apply Sigmoid activation",
  "helpUrl": ""
};
const rnn = {
  "type": "rnn_layer",
  "message0": "RNN layer with %1 units return sequences %2",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": 50,
      "min": 1
    },
    {
      "type": "field_checkbox",
      "name": "RETURN_SEQ",
      "checked": true
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Add a recurrent neural network layer",
  "helpUrl": ""
};
const dataselectionBlock = {
  "type": "dataset_selection",
  "message0": "Select dataset %1",
  "args0": [
    {
      "type": "field_dropdown",
      "name": "DATASET",
      "options": [
        ["MNIST", "MNIST"],
        ["CIFAR-10", "CIFAR10"],
        ["Custom path", "CUSTOM"]
      ]
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "output": "Dataset",
  "colour": 230,
  "tooltip": "Select a dataset to load",
  "helpUrl": ""
};
const dataLoaderBlock = {
  "type": "data_loader_config",
  "message0": "Load data with batch size %1 shuffle %2 workers %3",
  "args0": [
    {
      "type": "field_number",
      "name": "BATCH_SIZE",
      "value": 32,
      "min": 1
    },
    {
      "type": "field_checkbox",
      "name": "SHUFFLE",
      "checked": true
    },
    {
      "type": "field_number",
      "name": "WORKERS",
      "value": 4,
      "min": 0
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Configure data loader parameters",
  "helpUrl": ""
};

const dataPreprocessingBlock = {
  "type": "data_preprocessing",
  "message0": "Preprocess data with %1",
  "args0": [
    {
      "type": "field_input",
      "name": "METHOD",
      "text": "normalize"  // Default preprocessing method
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 160,
  "tooltip": "Apply preprocessing to data",
  "helpUrl": ""
};
const dataBatchingBlock = {
  "type": "data_batching",
  "message0": "Create batches of size %1",
  "args0": [
    {
      "type": "field_number",
      "name": "BATCH_SIZE",
      "value": 32,  // Default batch size
      "min": 1
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 160,
  "tooltip": "Create data batches",
  "helpUrl": ""
};
const dataShufflingBlock = {
  "type": "data_shuffling",
  "message0": "Shuffle data",
  "previousStatement": null,
  "nextStatement": null,
  "colour": 160,
  "tooltip": "Shuffle the dataset",
  "helpUrl": ""
};
const transformationsBlock = {
  "type": "data_transformations",
  "message0": "Apply transformations %1",
  "args0": [
    {
      "type": "input_statement",
      "name": "TRANSFORMS"
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Add transformations to the data loading process",
  "helpUrl": ""
};
const splitDataBlock = {
  "type": "split_data",
  "message0": "Split data into train %1 %, validation %2 %, test %3 %",
  "args0": [
    {
      "type": "field_number",
      "name": "TRAIN",
      "value": 70,
      "min": 0,
      "max": 100
    },
    {
      "type": "field_number",
      "name": "VALID",
      "value": 15,
      "min": 0,
      "max": 100
    },
    {
      "type": "field_number",
      "name": "TEST",
      "value": 15,
      "min": 0,
      "max": 100
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Split the dataset into training, validation, and test sets",
  "helpUrl": ""
};





// Create the block definitions for the JSON-only blocks.
// This does not register their definitions with Blockly.
// This file has no side effects!
export const blocks = Blockly.common.createBlockDefinitionsFromJsonArray(
  [python_function,python_class,DataWrapperG, addText, AddVectors, generateRandome, flattenLayer, denseLayer, maxPoolLayer,relu,conv,self,batchNorm,averagePool,dropout,tanh,sigmoid,rnn,dataBatchingBlock,dataLoaderBlock,dataselectionBlock,dataPreprocessingBlock,dataShufflingBlock,
    transformationsBlock,splitDataBlock
  ]
);


// Create the block definitions for the JSON-only blocks.
// This does not register their definitions with Blockly.
// This file has no side effects!
