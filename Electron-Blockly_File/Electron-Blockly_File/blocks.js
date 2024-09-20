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
  "tooltip": "Define a Python function with methods and attributes.",
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
  "previousStatement": null,
  "nextStatement": null,

  "tooltip": "Define a Python class with methods",
  "helpUrl": ""
};
const python_loop = {
  "type": "python_loop",
  "message0": "For each %1 in  %2 :%3 %4",
  "args0": [
    {
      "type": "field_input",
      "name": "element",
      "text": ""
    },
    {
      "type": "field_input",
      "name": "FIELD",
      "text": ""
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
  "tooltip": "Create a simpy loop to iterate through an field.",
  "helpUrl": ""
};
const nn_compact= {
  "type": "nn_compact",
  "message0": "@nn.compact",
  "colour": 'A52A2A',
  "previousStatement": null,
  "nextStatement": null,
  "tooltip": "Define a Flax @nn.compact decorater",
  "helpUrl": ""
};
const cheatblock= {
  "type": "cheatblock",
  "message0": " %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  'output': 'Number',
  "colour": "#B0BEC5",
  "previousStatement": null,
  "nextStatement": null,
  "tooltip": "Line of code which couldn't be translated by FlaxBlocks. These blocks are created when only a few Blocks are missing for a full scope translation",
  "helpUrl": ""
};

const nn_nowrap= {
  "type": "nn_nowrap",
  "message0": "@nn.nowrap",
  "colour": 'A52A2A',
  "previousStatement": null,
  "nextStatement": null,
  "tooltip": "Define a Flax @nn.nowrap decorater",
  "helpUrl": ""
};
const python_class_attribute = {
  "type": "python_class_attribute",
  "message0": "attribute %1 : %2",
  "args0": [
    {
      "type": "field_input",
      "name": "ATTRIBUTE_NAME",
      "text": "latents"
    },
    {
      "type": "field_input",
      "name": "ATTRIBUTE_VALUE",
      "text": "20"
    }
  ],
  "colour": 160,
  "previousStatement": null,
  "nextStatement": null,
  "tooltip": "Define a Python class attribute.",
  "helpUrl": ""
};
const python_return = {
  "type": "python_return",
  "message0": "return %1",
  "args0": [
    {
      "type": "field_input",
      "name": "RETURN_VALUE",
      "text": "value"
    }
  ],
  "colour": 160,
  "previousStatement": null,
  "nextStatement": null,
  "tooltip": "Return a value from a function.",
  "helpUrl": ""
};


const add_text = {
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

const Add_Vectors = {
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
const generate_randomePRNG = {
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




const set_var = {
  "type": "set_var",
  "message0": "Set %1 to %2",
  "args0": [
    {
      "type":  "field_input",
      "name": "SET_VARIABLE",
      "variable": "item"
    },
    {
      "type": "input_value",
      "name": "VALUE"
    }
  ],
  "output": null,
  "previousStatement": null,
  "nextStatement": null,
  "colour": 340,
  "tooltip": "Set a variable to a value",
  "helpUrl": ""
};


const get_variable = {
  "type": "get_variable",
  "message0": "Get %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VARIABLE_NAME",
      "variable": "item"
    }
  ],
  "output": null,
  "colour": 340,
  "tooltip": "Get the value of a variable",
  "helpUrl": ""
};


const celu_layer = {
  "type": "celu_layer",
  "message0": "CELU activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the CELU activation function.",
  "helpUrl": ""
};

const elu_layer = {
  "type": "elu_layer",
  "message0": "ELU activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the ELU activation function.",
  "helpUrl": ""
};

const gelu_layer = {
  "type": "gelu_layer",
  "message0": "GELU activation",
  "output":null,

  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Applies the GELU activation function.",
  "helpUrl": ""
};

const glu_layer = {
  "type": "glu_layer",
  "message0": "GLU activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the GLU activation function.",
  "helpUrl": ""
};

const hard_sigmoid_layer = {
  "type": "hard_sigmoid_layer",
  "message0": "Hard Sigmoid activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the Hard Sigmoid activation function.",
  "helpUrl": ""
};
const silu = {
  "type": "silu",
  "message0": "Silu activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the Silu activation function.",
  "helpUrl": ""
};

const hard_silu_layer = {
  "type": "hard_silu_layer",
  "message0": "Hard SiLU activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the Hard SiLU activation function.",
  "helpUrl": ""
};
const hard_tanh = {
  "type": "hard_tanh",
  "message0": "Hard Tanh activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the Hard Tanh activation function.",
  "helpUrl": ""
};
const leaky_relu = {
  "type": "leaky_relu",
  "message0": "Leaky Relu activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the Leaky Relu activation function.",
  "helpUrl": ""
};
const log_sigmoid = {
  "type": "log_sigmoid",
  "message0": "Log Sigmoid  activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the Log Sigmoid activation function.",
  "helpUrl": ""
};
const one_hot = {
  "type": "one_hot",
  "message0": "one hot activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the one hot activation function.",
  "helpUrl": ""
};
const selu = {
  "type": "selu",
  "message0": "selu activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the selu activation function.",
  "helpUrl": ""
};
const soft_sign = {
  "type": "soft_sign",
  "message0": "soft sign activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the soft sign activation function.",
  "helpUrl": ""
};
const softmax = {
  "type": "softmax",
  "message0": "softmax activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the softmax activation function.",
  "helpUrl": ""
};
const softplus = {
  "type": "softplus",
  "message0": "softplus activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the softplus activation function.",
  "helpUrl": ""
};
const standardize = {
  "type": "standardize",
  "message0": "standardize activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the standardize activation function.",
  "helpUrl": ""
};
const swish = {
  "type": "swish",
  "message0": "swish activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the swish activation function.",
  "helpUrl": ""
};
const log_softmax = {
  "type": "log_softmax",
  "message0": "Log softmax  activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the Log softmax activation function.",
  "helpUrl": ""
};
const logsumexp = {
  "type": "logsumexp",
  "message0": "Log sumexp  activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Applies the Log sumexp activation function.",
  "helpUrl": ""
};
const embed = {
  'type': 'Embed',
  'message0': 'Embed with parameters %1 input %2',
  'args0': [
    {
      'type': 'field_input',
      'name': 'PARAMS',
      'text': ''  // Users will input all parameters here
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 230,
  'tooltip': 'Adds an embedding layer with user-defined parameters.',
  'helpUrl': ''
};
const einsum = {
  'type': 'einsum',
  'message0': 'Einsum with parameters %1 input %2',
  'args0': [
    {
      'type': 'field_input',
      'name': 'PARAMS',
      'text': ''  // Users will input all parameters here
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 230,
  'tooltip': 'Adds an Einsum layer with user-defined parameters.',
  'helpUrl': ''
};

const scan = {
  'type': 'scan',
  'message0': 'Scan with parameters %1 units',
  'args0': [

    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 500,
  'tooltip': 'Applies the scan transformation with user-defined parameters.',
  'helpUrl': ''
};

const vmap = {
  'type': 'vmap',
  'message0': 'Vmap with parameters %1 units',
  'args0': [

    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 500,
  'tooltip': 'Applies the vmap transformation with user-defined parameters.',
  'helpUrl': ''
};
const remat_scan = {
  "type": "remat_scan",
  "message0": "remat_scan  with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 500,
  'tooltip': 'Applies the remat_scan compilation with user-defined parameters.',
  "helpUrl": ""
};
const jit = {
  'type': 'jit',
  'message0': 'Jit with parameters %1 unit',
  'args0': [

    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 500,
  'tooltip': 'Applies the JIT compilation with user-defined parameters.',
  'helpUrl': ''
};

const remat = {
  'type': 'remat',
  'message0': 'Remat with parameters %1 unit',
  'args0': [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 500,
  'tooltip': 'Applies the remat transformation with user-defined parameters.',
  'helpUrl': ''
};
const map_variables = {
  "type": "map_variables",
  "message0": "map_variables with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 500,
  "tooltip": "Applies the map_variables compilation with user-defined parameters.",
  "helpUrl": ""
};
const jvp = {
  'type': 'jvp',
  'message0': 'jvp with parameters %1 unit',
  'args0': [

    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 500,
  'tooltip': 'Applies the jvp compilation with user-defined parameters.',
  'helpUrl': ''
};

const vjp = {
  'type': 'vjp',
  'message0': 'vjp with parameters %1 unit',
  'args0': [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 500,
  'tooltip': 'Applies the vjp transformation with user-defined parameters.',
  'helpUrl': ''
};
const custom_vjp = {
  "type": "custom_vjp",
  "message0": "custom_vjp with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 500,
  "tooltip": "Applies the custom_vjp compilation with user-defined parameters.",
  "helpUrl": ""
};
const while_loop = {
  'type': 'while_loop',
  'message0': 'while_loop with parameters %1 unit',
  'args0': [

    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 500,
  'tooltip': 'Applies the while_loop compilation with user-defined parameters.',
  'helpUrl': ''
};

const cond = {
  'type': 'cond',
  'message0': 'cond with parameters %1 unit',
  'args0': [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 500,
  'tooltip': 'Applies the cond transformation with user-defined parameters.',
  'helpUrl': ''
};

const tabulate = {
  'type': 'tabulate',
  'message0': 'Tabulate with parameters %1 input %2',
  'args0': [
    {
      'type': 'field_input',
      'name': 'PARAMS',
      'text': ''  // Users input all parameters here
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 270,
  'tooltip': 'Applies the tabulate function with user-defined parameters.',
  'helpUrl': ''
};

const switchblock = {
  "type": "switchblock",
  "message0": "switch with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 500,
  "tooltip": "Applies the switch compilation with user-defined parameters.",
  "helpUrl": ""
};

const gelu = {
  "type": "gelu",
  "message0": "nn.gelu() activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "previousStatement": null,
  "nextStatement": null,
  "colour": 230,
  "tooltip": "Apply GELU activation",
  "helpUrl": ""
};

const reshape = {
  'type': 'reshape',
  'message0': 'Flatten tensor %1',
  'args0': [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 210,
  'tooltip': 'Flattens the input tensor into a single continuous vector.',
  'helpUrl': ''
};

const comment = {
  "type": "comment",
  "message0": "Add comment %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": 'Variable',
  "previousStatement": null,
  "nextStatement": null,
  "colour": 120,
  "tooltip": "Add a comment",
  "helpUrl": ""
};

const denseLayer = {
  "type": "Dense",
  "message0": "Dense layer with units %1 input %2",
  "args0": [
    {
      "type": "field_input",  // Changed to input_value
      "name": "UNITS",
      "check": "Number"  // Ensure this is validated as a number
    },
    {
      "type": "field_input",  // Changed to input_value
      "name": "VAR",
      "check": "Variable"  // Check to ensure the connected block is of type Variable
    }
  ],
  "output": "Variable",
  "colour": 230,
  "tooltip": "Adds a dense layer with the specified number of units.",
  "helpUrl": ""
};

const Sequential = {
  'type': 'Sequential',
  'message0': 'Sequential Combinator with units %1 input %2',
  'args0': [
    {
      'type': 'field_input',
      'name': 'UNITS',
      'text': '10'
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 300,
  'tooltip': 'Applies a linear chain of Modules.',
  'helpUrl': ''
};
const Variable = {
  "type": "Variable",
  "message0": "Variable with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 180,
  "tooltip": "A Variable object allows mutable access to a variable in a VariableDict.",
  "helpUrl": ""
};
const max_pool_layer = {
  'type': 'max_pool_layer',
  'message0': 'MaxPool with window shape %1 %2 strides %3 %4 input %5',
  'args0': [
    {
      'type': 'field_number',
      'name': 'WINDOW_SHAPE_X',
      'value': 2
    },
    {
      'type': 'field_number',
      'name': 'WINDOW_SHAPE_Y',
      'value': 2
    },
    {
      'type': 'field_number',
      'name': 'STRIDE_X',
      'value': 2
    },
    {
      'type': 'field_number',
      'name': 'STRIDE_Y',
      'value': 2
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 120,
  'tooltip': 'Applies a MaxPooling operation with specified window shape and strides.',
  'helpUrl': ''
};
const pool_layer = {
  'type': 'pool_layer',
  'message0': 'Pool with window shape %1 %2 strides %3 %4 input %5',
  'args0': [
    {
      'type': 'field_number',
      'name': 'WINDOW_SHAPE_X',
      'value': 2
    },
    {
      'type': 'field_number',
      'name': 'WINDOW_SHAPE_Y',
      'value': 2
    },
    {
      'type': 'field_number',
      'name': 'STRIDE_X',
      'value': 2
    },
    {
      'type': 'field_number',
      'name': 'STRIDE_Y',
      'value': 2
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 120,
  'tooltip': 'Helper function to define pooling functions.',
  'helpUrl': ''
};


const relu = {
  "type": "relu",
  "message0": "ReLU activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Apply ReLU activation",
  "helpUrl": ""
};
const PReLU = {
  "type": "PReLU",
  "message0": "PReLU activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Apply PReLU activation",
  "helpUrl": ""
};
const batch_norm_layer = {
  "type": "batch_norm_layer",
  "message0": "Batch Normalization %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 50,
  "tooltip": "Apply batch normalization",
  "helpUrl": ""
};
const layer_norm_layer = {
  "type": "layer_norm_layer",
  "message0": "Layer Normalization %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 50,
  "tooltip": "Apply Layer normalization",
  "helpUrl": ""
};
const group_norm_layer = {
  "type": "group_norm_layer",
  "message0": "Group Normalization %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 50,
  "tooltip": "Apply group normalization",
  "helpUrl": ""
};
const RMS_norm_layer = {
  "type": "RMS_norm_layer",
  "message0": "RMS Normalization %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 50,
  "tooltip": "Apply RMS normalization",
  "helpUrl": ""
};
const Instance_norm_layer = {
  "type": "Instance_norm_layer",
  "message0": "Instance Normalization %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 50,
  "tooltip": "Apply instance normalization",
  "helpUrl": ""
};
const Spectral_norm_layer = {
  "type": "Spectral_norm_layer",
  "message0": "Spectral Normalization %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 50,
  "tooltip": "Apply spectral normalization",
  "helpUrl": ""
};
const Weight_norm_layer = {
  "type": "Weight_norm_layer",
  "message0": "Weight Normalization %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 50,
  "tooltip": "Apply Weight normalization",
  "helpUrl": ""
};
const MultiHeadDotProductAttention = {
  "type": "MultiHeadDotProductAttention",
  "message0": "Multi-Head-Dot-Product Attention %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "Multi-Head-Dot-Product Attention",
  "helpUrl": ""
};
const MultiHeadAttention = {
  "type": "MultiHeadAttention",
  "message0": "Multi-Head Attention  %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "Multi-Head Attention",
  "helpUrl": ""
};
const SelfAttention = {
  "type": "SelfAttention",
  "message0": "Self Attention  %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "Self Attention",
  "helpUrl": ""
};
const DotProductAttention = {
  "type": "DotProductAttention",
  "message0": "Dot Product Attention  %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "Dot Product Attention",
  "helpUrl": ""
};
const DotProductAttentionWeights = {
  "type": "DotProductAttentionWeights",
  "message0": "Dot Product Attention Weights %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "Dot Product Attention Weights",
  "helpUrl": ""
};
const makecausalmask = {
  "type": "makecausalmask",
  "message0": "Make  causal mask  %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "Make a causal mask for self-attention.",
  "helpUrl": ""
};
const makeattentionmask = {
  "type": "makeattentionmask",
  "message0": "Make  Attention mask  %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "Mask-making helper for attention weights.",
  "helpUrl": ""
};


const avg_pool = {
  "type": "avg_pool",
  "message0": "Average Pool with pool size %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 120,
  "tooltip": "Apply an average pooling layer.",
  "helpUrl": ""
};

const Dropout = {
  "type": "Dropout",
  "message0": "Dropout with rate %1 input %2",
  "args0": [
    {
      "type": "field_number",
      "name": "RATE",
      "value": 0.5
    },
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 180,
  "tooltip": "Apply Dropout for regularization.",
  "helpUrl": ""
};

const tanh_layer = {
  "type": "tanh_layer",
  "message0": "Tanh activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Apply Tanh activation",
  "helpUrl": ""
};

const sigmoid_layer = {
  "type": "sigmoid_layer",
  "message0": "Sigmoid activation %1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Apply Sigmoid activation",
  "helpUrl": ""
};

const rnn_layer = {
  "type": "rnn_layer",
  "message0": "RNN with %1 units ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": " The RNN module takes any RNNCellBase instance and applies it over a sequence",
  "helpUrl": ""
};
const RNNCellBase = {
  "type": "RNNCellBase",
  "message0": "RNNCellBase layer with %1 units ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": " RNN cell Base",
  "helpUrl": ""
};
const LSTMCell = {
  "type": "LSTMCell",
  "message0": "LSTMCell layer with %1 units ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": "LSTM cell.",
  "helpUrl": ""
};
const OptimizedLSTMCell = {
  "type": "OptimizedLSTMCell",
  "message0": "Optimized LSTM Cell layer with %1 units ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": "More efficient LSTM Cell that concatenates state components before matmul.",
  "helpUrl": ""
};
const ConvLSTMCell = {
  "type": "ConvLSTMCell",
  "message0": "convolutional LSTM cell with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": "A convolutional LSTM cell.",
  "helpUrl": ""
};
const SimpleCell = {
  "type": "SimpleCell",
  "message0": "Simple Cell with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": "A Simple cell",
  "helpUrl": ""
};
const GRUCell = {
  "type": "GRUCell",
  "message0": "GRU Cell with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": "A GRU cell",
  "helpUrl": ""
};
const MGUCell = {
  "type": "MGUCell",
  "message0": "MGU Cell with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": "An MGU cell",
  "helpUrl": ""
};
const Bidirectional = {
  "type": "Bidirectional",
  "message0": "Bidirectional Cell with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 230,
  "tooltip": "Processes the input in both directions and merges the results.",
  "helpUrl": ""
};

const Dense_General = {
  'type': 'Dense_General',
  'message0': 'DenseGeneral with parameters %1 input %2',
  'args0': [
    {
      'type': 'field_input',
      'name': 'PARAMS',
      'text': ''  // Users input all parameters here
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 230,
  'tooltip': 'Adds a DenseGeneral layer with user-defined parameters.',
  'helpUrl': ''
};


const conv = {
  'type': 'Conv',
  'message0': 'Conv Layer with parameters %1 input %2',
  'args0': [
    {
      'type': 'field_input',
      'name': 'PARAMS',
      'text': ''  // Users will input all parameters here
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 230,
  'tooltip': 'Adds a Conv Layer layer with user-defined parameters.',
  'helpUrl': ''
};
const conv_Local = {
  'type': 'conv_local',
  'message0': 'conv_Local with parameters %1 input %2',
  'args0': [
    {
      'type': 'field_input',
      'name': 'PARAMS',
      'text': ''  // Users will input all parameters here
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 230,
  'tooltip': 'Adds a conv_Local layer with user-defined parameters.',
  'helpUrl': ''
};
const convTranspose = {
  'type': 'convTranspose',
  'message0': 'ConvTranspose with parameters %1 input %2',
  'args0': [
    {
      'type': 'field_input',
      'name': 'PARAMS',
      'text': ''  // Users will input all parameters here
    },
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  'output': 'Variable',
  'colour': 230,
  'tooltip': 'Adds a ConvTranspose layer with user-defined parameters.',
  'helpUrl': ''
};

const constant = {
  "type": "constant",
  "message0": "constant initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a constant uniform initializer.",
  "helpUrl": ""
};
const delta_orthogonal = {
  "type": "delta_orthogonal",
  "message0": "delta_orthogonal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a delta_orthogonal uniform initializer.",
  "helpUrl": ""
};
const glorot_normal = {
  "type": "glorot_normal",
  "message0": "glorot_normal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a glorot_normal uniform initializer.",
  "helpUrl": ""
};
const glorot_uniform = {
  "type": "glorot_uniform",
  "message0": "glorot_uniform initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a glorot_uniform uniform initializer.",
  "helpUrl": ""
};
const he_normal = {
  "type": "he_normal",
  "message0": "he_normal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a he_normal uniform initializer.",
  "helpUrl": ""
};const he_uniform = {
  "type": "he_uniform",
  "message0": "he_uniform initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a he_uniform uniform initializer.",
  "helpUrl": ""
};
const kaiming_normal = {
  "type": "kaiming_normal",
  "message0": "kaiming_normal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a kaiming_normal uniform initializer.",
  "helpUrl": ""
};
const lecun_normal = {
  "type": "lecun_normal",
  "message0": "lecun_normal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a lecun_normal uniform initializer.",
  "helpUrl": ""
};
const lecun_uniform = {
  "type": "lecun_uniform",
  "message0": "lecun_uniform initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a lecun_uniform uniform initializer.",
  "helpUrl": ""
};
const normal = {
  "type": "normal",
  "message0": "normal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a normal uniform initializer.",
  "helpUrl": ""
};
const kaiming_uniform = {
  "type": "kaiming_uniform",
  "message0": "kaiming_uniform initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a kaiming_uniform uniform initializer.",
  "helpUrl": ""
};
const truncated_normal = {
  "type": "truncated_normal",
  "message0": "truncated_normal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a truncated_normal uniform initializer.",
  "helpUrl": ""
};
const ones = {
  "type": "ones",
  "message0": "ones initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a ones uniform initializer.",
  "helpUrl": ""
};
const orthogonal = {
  "type": "orthogonal",
  "message0": "orthogonal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a orthogonal uniform initializer.",
  "helpUrl": ""
};
const xavier_uniform = {
  "type": "xavier_uniform",
  "message0": "xavier_uniform initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a xavier_uniform uniform initializer.",
  "helpUrl": ""
};
const xavier_normal = {
  "type": "xavier_normal",
  "message0": "xavier_normal initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a xavier_normal uniform initializer.",
  "helpUrl": ""
};
const ones_init = {
  "type": "ones_init",
  "message0": "ones_init initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a ones_init uniform initializer.",
  "helpUrl": ""
};
const uniform = {
  "type": "uniform",
  "message0": "uniform initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a uniform uniform initializer.",
  "helpUrl": ""
};
const variance_scaling = {
  "type": "variance_scaling",
  "message0": "variance_scaling initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a variance_scaling variance_scaling initializer.",
  "helpUrl": ""
};
const zeros_init = {
  "type": "zeros_init",
  "message0": "zeros_init initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a zeros_init zeros_init initializer.",
  "helpUrl": ""
};
const zeros = {
  "type": "zeros",
  "message0": "zeros initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 10,
  "tooltip": "Builds a zeros zeros initializer.",
  "helpUrl": ""
};
const apply = {
  "type": "apply",
  "message0": "apply initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 400,
  "tooltip": "Builds a apply apply initializer.",
  "helpUrl": ""
};
const init = {
  "type": "init",
  "message0": "init initializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 400,
  "tooltip": "Builds a init init initializer.",
  "helpUrl": ""
};
const init_with_output = {
  "type": "init_with_output",
  "message0": "init_with_output init_with_outputializer with %1 Unit ",
  "args0": [
    {
      "type": "field_number",
      "name": "UNITS",
      "value": "params"
    },
  ],
  "output": null,
  "colour": 400,
  "tooltip": "Builds a init_with_output init_with_output init_with_outputializer.",
  "helpUrl": ""
};
const Partitioned = {
  "type": "Partitioned",
  "message0": "Partitioned with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A Partitioned object allows mutable access to a Partitioned in a PartitionedDict.",
  "helpUrl": ""
};

const get_partition_spec = {
  "type": "get_partition_spec",
  "message0": "get_partition_spec with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A get_partition_spec object allows mutable access to a get_partition_spec in a get_partition_specDict.",
  "helpUrl": ""
};
const get_sharding = {
  "type": "get_sharding",
  "message0": "get_sharding with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A get_sharding object allows mutable access to a get_sharding in a get_shardingDict.",
  "helpUrl": ""
};
const LogicallyPartitioned = {
  "type": "LogicallyPartitioned",
  "message0": "LogicallyPartitioned with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A LogicallyPartitioned object allows mutable access to a LogicallyPartitioned in a LogicallyPartitionedDict.",
  "helpUrl": ""
};
const logical_axis_rules = {
  "type": "logical_axis_rules",
  "message0": "logical_axis_rules with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A logical_axis_rules object allows mutable access to a logical_axis_rules in a logical_axis_rulesDict.",
  "helpUrl": ""
};
const logical_to_mesh_axes = {
  "type": "logical_to_mesh_axes",
  "message0": "logical_to_mesh_axes with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A logical_to_mesh_axes object allows mutable access to a logical_to_mesh_axes in a logical_to_mesh_axesDict.",
  "helpUrl": ""
};
const set_logical_axis_rules = {
  "type": "set_logical_axis_rules",
  "message0": "set_logical_axis_rules with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A set_logical_axis_rules object allows mutable access to a set_logical_axis_rules in a set_logical_axis_rulesDict.",
  "helpUrl": ""
};
const get_logical_axis_rules = {
  "type": "get_logical_axis_rules",
  "message0": "get_logical_axis_rules with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A get_logical_axis_rules object allows mutable access to a get_logical_axis_rules in a get_logical_axis_rulesDict.",
  "helpUrl": ""
};
const logical_to_mesh_sharding = {
  "type": "logical_to_mesh_sharding",
  "message0": "logical_to_mesh_sharding with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A logical_to_mesh_sharding object allows mutable access to a logical_to_mesh_sharding in a logical_to_mesh_shardingDict.",
  "helpUrl": ""
};
const with_logical_constraint = {
  "type": "with_logical_constraint",
  "message0": "with_logical_constraint with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A with_logical_constraint object allows mutable access to a with_logical_constraint in a with_logical_constraintDict.",
  "helpUrl": ""
};
const with_logical_partitioning = {
  "type": "with_logical_partitioning",
  "message0": "with_logical_partitioning with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A with_logical_partitioning object allows mutable access to a with_logical_partitioning in a with_logical_partitioningDict.",
  "helpUrl": ""
};
const with_partitioning = {
  "type": "with_partitioning",
  "message0": "with_partitioning with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A with_partitioning object allows mutable access to a with_partitioning in a with_partitioningDict.",
  "helpUrl": ""
};
const logical_to_mesh = {
  "type": "logical_to_mesh",
  "message0": "logical_to_mesh with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": 70,
  "tooltip": "A logical_to_mesh object allows mutable access to a logical_to_mesh in a logical_to_meshDict.",
  "helpUrl": ""
};
const enable_named_call = {
  "type": "enable_named_call",
  "message0": "enable_named_call with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": '#FFA500',
  "tooltip": "A enable_named_call object allows mutable access to a enable_named_call in a enable_named_callDict.",
  "helpUrl": ""
};
const disable_named_call = {
  "type": "disable_named_call",
  "message0": "disable_named_call with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": '#FFA500',
  "tooltip": "A disable_named_call object allows mutable access to a disable_named_call in a disable_named_callDict.",
  "helpUrl": ""
};
const override_named_call = {
  "type": "override_named_call",
  "message0": "override_named_call with %1 Unit ",
  "args0": [
    {
      'type': 'field_input',
      'name': 'VAR',
      'check': 'Variable'
    }
  ],
  "output": null,
  "colour": '#FFA500',
  "tooltip": "A override_named_call object allows mutable access to a override_named_call in a override_named_callDict.",
  "helpUrl": ""
};
const string = {
  "type": "string",
  "message0": "%1",
  "args0": [
    {
      "type": "field_input",
      "name": "VAR",
      "variable": "Variable"
    }
  ],
  'output': 'Number',
  'colour': 130,
  'tooltip': 'Create a normal string with desired parameter',
  'helpUrl': ''
}
const  member = {
 "type": "member",
  "message0": " %1 to %2",
  "args0": [
    {
      "type":  "field_input",
      "name": "SET_VARIABLE",
      "variable": "item"
    },
    {
      "type": "input_value",
      "name": "VALUE"
    }
  ],

  "previousStatement": null,
  "nextStatement": null,
  "colour": 340,
  "tooltip": "Set a variable to a value",
  "helpUrl": ""
}
const  Declaration = {
  "type": "Declaration",
  "message0": " %1",
  "args0": [
    {
      "type":  "field_input",
      "name": "SET_VARIABLE",
      "variable": "item"
    },

  ],

  "previousStatement": null,
  "nextStatement": null,
  "colour": 340,
  "tooltip": "Create a Declaration",
  "helpUrl": ""
}
// Add the new blocks to the blocks array
export const blocks = Blockly.common.createBlockDefinitionsFromJsonArray([ constant,delta_orthogonal,glorot_normal,glorot_uniform,he_normal,he_uniform,
  celu_layer,elu_layer,gelu_layer,glu_layer,hard_sigmoid_layer,hard_silu_layer,embed,scan,vmap,tabulate,kaiming_normal,lecun_normal,
  lecun_uniform,normal,kaiming_uniform, truncated_normal,ones,orthogonal,xavier_uniform,xavier_normal,ones_init,uniform,variance_scaling,enable_named_call,
  ,gelu,python_function, python_class, DataWrapperG, add_text, Add_Vectors, generate_randomePRNG,conv,conv_Local,convTranspose,zeros_init,zeros,disable_named_call,
  reshape, denseLayer, max_pool_layer, relu, self, batch_norm_layer,layer_norm_layer,group_norm_layer,RMS_norm_layer,Instance_norm_layer,Spectral_norm_layer,Weight_norm_layer,init,nn_nowrap,
  avg_pool, Dropout, tanh_layer, sigmoid_layer, rnn_layer,softmax,softplus,standardize,logsumexp,one_hot,selu,silu,soft_sign,apply,init_with_output,override_named_call,
  python_class_attribute, python_return, nn_compact, set_var, set_var,Dense_General,jit,remat,einsum,pool_layer,swish,get_variable,
  Sequential,MultiHeadAttention,MultiHeadDotProductAttention,SelfAttention,DotProductAttention,DotProductAttentionWeights,makecausalmask,makeattentionmask,
  RNNCellBase,LSTMCell,OptimizedLSTMCell,vjp,ConvLSTMCell,SimpleCell,GRUCell,MGUCell,Bidirectional,comment,relu,PReLU, hard_tanh,leaky_relu,log_sigmoid,log_softmax
  ,remat_scan,map_variables,switchblock,jvp,custom_vjp,while_loop,cond,Variable,Partitioned,with_partitioning,get_partition_spec,get_sharding,LogicallyPartitioned,
  logical_axis_rules,set_logical_axis_rules,get_logical_axis_rules,logical_to_mesh_axes,logical_to_mesh,logical_to_mesh_sharding,with_logical_constraint,with_logical_partitioning,python_loop,string,member,Declaration,cheatblock
]);










