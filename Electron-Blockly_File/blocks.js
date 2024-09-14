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
const nn_compact= {
  "type": "nn_compact",
  "message0": "@nn.compact",
  "colour": 160,
  "previousStatement": null,
  "nextStatement": null,
  "tooltip": "Define a Flax @nn.compact decorater",
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

const lossFunctionBlock = {
  "type": "loss_function",
  "message0": "Loss function %1",
  "args0": [
    {
      "type": "field_dropdown",
      "name": "LOSS_FUNCTION",
      "options": [
        ["Mean Squared Error", "mse"],
        ["Cross Entropy", "cross_entropy"],
        ["Mean Absolute Error", "mae"]
      ]
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 160,
  "tooltip": "Define a loss function",
  "helpUrl": ""
};

// Optimizer Block
const optimizerBlock = {
  "type": "optimizer",
  "message0": "Optimizer %1 with learning rate %2",
  "args0": [
    {
      "type": "field_dropdown",
      "name": "OPTIMIZER",
      "options": [
        ["Adam", "adam"],
        ["SGD", "sgd"],
        ["RMSprop", "rmsprop"]
      ]
    },
    {
      "type": "field_number",
      "name": "LEARNING_RATE",
      "value": 0.001,
      "min": 0,
      "precision": 0.0001
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 160,
  "tooltip": "Define an optimizer",
  "helpUrl": ""
};

// Training Step Block
const trainingStepBlock = {
  "type": "training_step",
  "message0": "Training step with model %1 data %2 loss %3 optimizer %4",
  "args0": [
    {
      "type": "input_value",
      "name": "MODEL",
      "check": "String"
    },
    {
      "type": "input_value",
      "name": "DATA",
      "check": "String"
    },
    {
      "type": "input_value",
      "name": "LOSS",
      "check": "String"
    },
    {
      "type": "input_value",
      "name": "OPTIMIZER",
      "check": "String"
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 160,
  "tooltip": "Define a single training step",
  "helpUrl": ""
};

// Evaluation Block
const evaluationBlock = {
  "type": "evaluation",
  "message0": "Evaluate model %1 with data %2",
  "args0": [
    {
      "type": "input_value",
      "name": "MODEL",
      "check": "String"
    },
    {
      "type": "input_value",
      "name": "DATA",
      "check": "String"
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 160,
  "tooltip": "Evaluate the model",
  "helpUrl": ""
};

// Training Loop Block
const trainingLoopBlock = {
  "type": "training_loop",
  "message0": "Training loop for %1 epochs %2",
  "args0": [
    {
      "type": "input_statement",
      "name": "TRAINING_STEP"
    },
    {
      "type": "field_number",
      "name": "EPOCHS",
      "value": 10,
      "min": 1
    }
  ],
  "previousStatement": null,
  "nextStatement": null,
  "colour": 160,
  "tooltip": "Define the training loop",
  "helpUrl": ""
};

const setVariableBlock = {
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
  "colour": 230,
  "tooltip": "Set a variable to a value",
  "helpUrl": ""
};


const getVariableBlock = {
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
  "colour": 230,
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
  'type': 'embed',
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

const scanBlock = {
  'type': 'scan',
  'message0': 'Scan with parameters %1 input %2',
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
  'tooltip': 'Applies the scan transformation with user-defined parameters.',
  'helpUrl': ''
};

const vmapBlock = {
  'type': 'vmap',
  'message0': 'Vmap with parameters %1 input %2',
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
  'tooltip': 'Applies the vmap transformation with user-defined parameters.',
  'helpUrl': ''
};

const tabulateBlock = {
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
  'colour': 230,
  'tooltip': 'Applies the tabulate function with user-defined parameters.',
  'helpUrl': ''
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

const flattenLayer = {
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
      "type": "input_value",  // Changed to input_value
      "name": "UNITS",
      "check": "Number"  // Ensure this is validated as a number
    },
    {
      "type": "input_value",  // Changed to input_value
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

const maxPoolLayer = {
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
const poolLayer = {
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
const dump = {
  "type": "relu_layer",
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
}

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
const batchNorm = {
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
const layernorm = {
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
const groupnorm = {
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
const RMSNorm = {
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
const InstanceNorm = {
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
const SpectralNorm = {
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
const WeightNorm = {
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


const averagePool = {
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

const dropout = {
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
  "tooltip": "Apply dropout for regularization.",
  "helpUrl": ""
};

const tanh = {
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

const sigmoid = {
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

const denseGeneralBlock = {
  'type': 'Dense',
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

const jitBlock = {
  'type': 'jit',
  'message0': 'Jit with parameters %1 input %2',
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
  'tooltip': 'Applies the JIT compilation with user-defined parameters.',
  'helpUrl': ''
};

const rematBlock = {
  'type': 'remat',
  'message0': 'Remat with parameters %1 input %2',
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
  'tooltip': 'Applies the remat transformation with user-defined parameters.',
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
const convLocal = {
  'type': 'conv_local',
  'message0': 'ConvLocal with parameters %1 input %2',
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
  'tooltip': 'Adds a ConvLocal layer with user-defined parameters.',
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
  'output': 'Variable',
  'colour': 130,
  'tooltip': 'Create a normal string with desired parameter',
  'helpUrl': ''
}



// Add the new blocks to the blocks array
export const blocks = Blockly.common.createBlockDefinitionsFromJsonArray([
  celu_layer,elu_layer,gelu_layer,glu_layer,hard_sigmoid_layer,hard_silu_layer,embed,scanBlock,vmapBlock,tabulateBlock
  ,gelu,python_function, python_class, DataWrapperG, addText, AddVectors, generateRandome,conv,convLocal,convTranspose,
  flattenLayer, denseLayer, maxPoolLayer, relu, self, batchNorm,layernorm,groupnorm,RMSNorm,InstanceNorm,SpectralNorm,WeightNorm, averagePool, dropout, tanh, sigmoid, rnn_layer,
  dataBatchingBlock, dataLoaderBlock, dataselectionBlock, dataPreprocessingBlock, dataShufflingBlock, transformationsBlock,softmax,softplus,standardize,
  splitDataBlock, lossFunctionBlock, optimizerBlock, trainingStepBlock, trainingLoopBlock, evaluationBlock,logsumexp,one_hot,selu,silu,soft_sign,
  python_class_attribute, python_return, nn_compact, setVariableBlock, getVariableBlock,denseGeneralBlock,jitBlock,rematBlock,einsum,poolLayer,swish,
  Sequential,MultiHeadAttention,MultiHeadDotProductAttention,SelfAttention,DotProductAttention,DotProductAttentionWeights,makecausalmask,makeattentionmask,
  RNNCellBase,LSTMCell,OptimizedLSTMCell,ConvLSTMCell,SimpleCell,GRUCell,MGUCell,Bidirectional,comment,dump,PReLU, hard_tanh,leaky_relu,log_sigmoid,log_softmax,string

]);






















