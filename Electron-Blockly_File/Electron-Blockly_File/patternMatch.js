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
    'type': 'ConvTranspose',
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
    'output': 'Number',
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
    RNNCellBase,LSTMCell,OptimizedLSTMCell,ConvLSTMCell,SimpleCell,GRUCell,MGUCell,Bidirectional,comment,dump,PReLU, hard_tanh,leaky_relu,log_sigmoid,log_softmax,string,python_loop

]);

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



pythonGenerator.forBlock['self'] = function(block, generator) {
    var model = generator.valueToCode(block, 'func', Order.NONE);
    // TODO: Assemble python into code variable.
    // TODO: Change ORDER_NONE to the correct strength.
    return '@nn.compact\n' +
        'def __call__(self, x):\n';
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




pythonGenerator.forBlock['dataset_selection'] = function(block) {
    const dataset = block.getFieldValue('DATASET');
    return `dataset = load_dataset('${dataset}')\n`;
};
pythonGenerator.forBlock['data_loader_config'] = function(block) {
    const batchSize = block.getFieldValue('BATCH_SIZE');
    const shuffle = block.getFieldValue('SHUFFLE') === 'TRUE' ? 'True' : 'False';
    const workers = block.getFieldValue('WORKERS');
    return `data.DataLoader(dataset, batch_size=${batchSize}, shuffle=${shuffle}, num_workers=${workers})\n`;
};


pythonGenerator.forBlock['data_preprocessing'] = function(block) {
    const method = block.getFieldValue('METHOD');
    return `dataset = preprocess_data(dataset, method='${method}')\n`;
};

pythonGenerator.forBlock['data_batching'] = function(block) {
    const batchSize = block.getFieldValue('BATCH_SIZE');
    return `data_loader = DataLoader(dataset, batch_size=${batchSize}, shuffle=True)\n`;
};

pythonGenerator.forBlock['data_shuffling'] = function(block) {
    return `dataset = shuffle_data(dataset)\n`;
};
pythonGenerator.forBlock['data_transformations'] = function(block) {
    const statements_transforms = pythonGenerator.statementToCode(block, 'TRANSFORMS');
    return `transforms.Compose([\n${statements_transforms}])\n`;
};
pythonGenerator.forBlock['split_data'] = function(block) {
    const train = block.getFieldValue('TRAIN');
    const valid = block.getFieldValue('VALID');
    const test = block.getFieldValue('TEST');
    return `train_set, valid_set, test_set = split_dataset(dataset, train=${train}, validation=${valid}, test=${test})\n`;
};
pythonGenerator.forBlock['loss_function'] = function(block) {
    const lossFunction = block.getFieldValue('LOSS_FUNCTION');
    return `loss_fn = ${lossFunction}\n`;
};

pythonGenerator.forBlock['optimizer'] = function(block) {
    const optimizer = block.getFieldValue('OPTIMIZER');
    const learningRate = block.getFieldValue('LEARNING_RATE');
    return `optimizer = ${optimizer}(learning_rate=${learningRate})\n`;
};

pythonGenerator.forBlock['training_step'] = function(block) {
    const model = pythonGenerator.valueToCode(block, 'MODEL', Order.NONE);
    const data = pythonGenerator.valueToCode(block, 'DATA', Order.NONE);
    const loss = pythonGenerator.valueToCode(block, 'LOSS', Order.NONE);
    const optimizer = pythonGenerator.valueToCode(block, 'OPTIMIZER', Order.NONE);
    return `def train_step(model, data, loss_fn, optimizer):\n` +
        `    # Forward pass\n` +
        `    predictions = model(data)\n` +
        `    loss = loss_fn(predictions, data['labels'])\n` +
        `    # Backward pass and optimization\n` +
        `    optimizer.zero_grad()\n` +
        `    loss.backward()\n` +
        `    optimizer.step()\n`;
};

pythonGenerator.forBlock['evaluation'] = function(block) {
    const model = pythonGenerator.valueToCode(block, 'MODEL', Order.NONE);
    const data = pythonGenerator.valueToCode(block, 'DATA', Order.NONE);
    return `def evaluate_model(model, data):\n` +
        `    model.eval()\n` +
        `    with torch.no_grad():\n` +
        `        predictions = model(data)\n` +
        `        accuracy = (predictions.argmax(dim=1) == data['labels']).float().mean()\n` +
        `    return accuracy\n`;
};

pythonGenerator.forBlock['training_loop'] = function(block) {
    const trainingStep = pythonGenerator.statementToCode(block, 'TRAINING_STEP');
    const epochs = block.getFieldValue('EPOCHS');
    return `for epoch in range(${epochs}):\n` +
        `    ${trainingStep}\n` +
        `    print(f'Epoch {epoch+1}/{${epochs}} completed')\n`;
};
pythonGenerator.forBlock['python_class'] = function(block) {
    const className = block.getFieldValue('CLASS_NAME');
    const methods = pythonGenerator.statementToCode(block, 'METHODS');
    return `class ${className}:\n${methods}`;
};

pythonGenerator.forBlock['python_function'] = function(block) {
    const functionName = block.getFieldValue('CLASS_NAME');
    let methods = pythonGenerator.statementToCode(block, 'METHODS');
    if (methods) {
        methods = pythonGenerator.prefixLines(methods, pythonGenerator.INDENT);
    }
    return `def ${functionName}:\n${methods}`;
};

pythonGenerator.forBlock['python_loop'] = function(block) {
    const iterator = block.getFieldValue('element')
    const field = block.getFieldValue('FIELD');
    let methods = pythonGenerator.statementToCode(block, 'METHODS');
    if (methods) {
        methods = pythonGenerator.prefixLines(methods, pythonGenerator.INDENT);
    }
    return `for ${iterator} in ${field}:\n${methods}`;
};
pythonGenerator.forBlock['python_class_attribute'] = function(block) {
    const attributeName = block.getFieldValue('ATTRIBUTE_NAME');
    const attributeValue = block.getFieldValue('ATTRIBUTE_VALUE');
    return `${attributeName}: ${attributeValue}\n`;
};

pythonGenerator.forBlock['python_return'] = function(block) {
    const returnValue = block.getFieldValue('RETURN_VALUE');
    return `return ${returnValue}\n`;
};

pythonGenerator.forBlock['nn_compact'] = function(block) {
    return `@nn.compact\n`;
};
pythonGenerator.forBlock['set_var'] = function(block) {
    var variable = block.getFieldValue('SET_VARIABLE');
    const value = pythonGenerator.valueToCode(block, 'VALUE', Order.NONE);
    return `${variable} = ${value}\n`;
    //return [${variable} = ${value},Order.ATOMIC]
};
pythonGenerator.forBlock['get_variable'] = function(block) {
    const variableName = block.getFieldValue('VARIABLE_NAME');
    const code = `${variableName}`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['relu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.relu(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['PReLU'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.PReLU(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['hard_tanh'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.hard_tanh(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['leaky_relu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.leaky_relu(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['log_sigmoid'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.log_sigmoid(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['log_softmax'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.log_softmax(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['soft_sign'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.soft_sign(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['softmax'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.softmax(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['softplus'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.softplus(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['standardize'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.standardize(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['swish'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.swish(${variableName})\n`;
    return [code, Order.ATOMIC];
};

pythonGenerator.forBlock['logsumexp'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.logsumexp(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['one_hot'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.one_hot(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['selu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.one_hot(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['celu_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.celu(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['elu_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.elu(${variableName})`;
    return [code, Order.ATOMIC];
};

pythonGenerator.forBlock['gelu_layer'] = function(block) {
    const code = `nn.gelu(x)\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['glu_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.glu(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['hard_sigmoid_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.hard_sigmoid(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['hard_silu_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.hard_silu(${variableName})\n`;
    return [code, Order.ATOMIC];

};


pythonGenerator.forBlock['embed'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.Embed(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['einsum'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.Einsum(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['scan'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.scan(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['vmap'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.vmap(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['tabulate'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.tabulate(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['gelu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.gelu${variableName}\n`;
    return `nn.gelu${variableName}\n`;

};
pythonGenerator.forBlock['comment'] = function(block, generator) {
    const text = block.getFieldValue( 'VAR');

    return `# ${text}\n`;

};


pythonGenerator.forBlock['reshape'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `${variableName}.reshape((${variableName}.shape[0], -1))\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Dense'] = function(block, generator) {
    const units = block.getFieldValue('PARAMS') || '(x)';
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.Dense${units}${variableName}\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Sequential'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.Sequential(${units})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['max_pool_layer'] = function(block, generator) {
    const windowShapeX = block.getFieldValue('WINDOW_SHAPE_X');
    const windowShapeY = block.getFieldValue('WINDOW_SHAPE_Y');
    const strideX = block.getFieldValue('STRIDE_X');
    const strideY = block.getFieldValue('STRIDE_Y');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.max_pool(window_shape=(${windowShapeX}, ${windowShapeY}), strides=(${strideX}, ${strideY}))(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['pool_layer'] = function(block, generator) {
    const windowShapeX = block.getFieldValue('WINDOW_SHAPE_X');
    const windowShapeY = block.getFieldValue('WINDOW_SHAPE_Y');
    const strideX = block.getFieldValue('STRIDE_X');
    const strideY = block.getFieldValue('STRIDE_Y');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.pool(window_shape=(${windowShapeX}, ${windowShapeY}), strides=(${strideX}, ${strideY}))(${variableName})\n`;
    return [code, Order.ATOMIC];
};


pythonGenerator.forBlock['avg_pool'] = function(block, generator) {

    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.avg_pool${variableName}\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['Dropout'] = function(block, generator) {
    const rate = block.getFieldValue('RATE');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.Dropout(rate=${rate})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['batch_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.BatchNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['layer_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.LayerNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['group_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.GroupNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['RMS_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.RMSNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Instance_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.InstanceNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Spectral_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.SpectralNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Weight_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.WeightNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['MultiHeadAttention'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.MultiHeadAttention()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['MultiHeadDotProductAttention'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.MultiHeadDotProductAttention()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['SelfAttention'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.SelfAttention()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['DotProductAttention'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.dot_product_attention()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['DotProductAttentionWeights'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.dot_product_attention_weights()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['makeattentionmask'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.make_attention_mask()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['makecausalnmask'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.make_causal_mask()(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['tanh_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.tanh(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['sigmoid_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.sigmoid(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['silu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.silu(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['rnn_layer'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.RNN(units=${units})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['RNNCellBase'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.RNNCellBase(units=${units})\n`;
    return [code, Order.ATOMIC];

};pythonGenerator.forBlock['LSTMCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.LSTMCell(units=${units})\n`;
    return [code, Order.ATOMIC];

};pythonGenerator.forBlock['OptimizedLSTMCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.OptimizedLSTMCell(units=${units})\n`;
    return [code, Order.ATOMIC];

};pythonGenerator.forBlock['ConvLSTMCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.ConvLSTMCell(units=${units})\n`;
    return [code, Order.ATOMIC];

};pythonGenerator.forBlock['SimpleCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.SimpleCell(units=${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['GRUCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.GRUCell(units=${units})\n`;
    return [code, Order.ATOMIC];

};pythonGenerator.forBlock['MGUCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.MGUCell(units=${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Bidirectional'] = function(block, generator) {
    const units = block.getFieldValue('UNITS');
    const code = `nn.Bidirectional(units=${units})\n`;
    return [code, Order.ATOMIC];

};


pythonGenerator.forBlock['dense_general'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.DenseGeneral(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['jit'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.jit(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['remat'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.remat(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['convTranspose'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.ConvTranspose(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['conv_local'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.ConvLocal(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Conv'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') || 'x';
    const code = `nn.Conv(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['string'] = function(block, generator) {

    const text = block.getFieldValue( 'VAR');

    return [text, Order.ATOMIC];

};

export const toolbox = {
    'kind': 'categoryToolbox',
    'contents': [
        {
            'kind': 'category',
            'name': 'Logic',
            'categorystyle': 'logic_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'controls_if',
                },
                {
                    'kind': 'block',
                    'type': 'logic_compare',
                },
                {
                    'kind': 'block',
                    'type': 'logic_operation',
                },
                {
                    'kind': 'block',
                    'type': 'logic_negate',
                },
                {
                    'kind': 'block',
                    'type': 'logic_boolean',
                },
                {
                    'kind': 'block',
                    'type': 'logic_null',
                },
                {
                    'kind': 'block',
                    'type': 'logic_ternary',
                },
            ],
        },
        {
            'kind': 'category',
            'name': 'Classes',
            'categorystyle': 'procedure_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'python_class'
                },
                {
                    'kind': 'block',
                    'type': 'python_function'
                },
                {
                    'kind': 'block',
                    'type': 'python_loop'
                },
                {
                    'kind': 'block',
                    'type': 'python_class_attribute'
                },
                {
                    'kind': 'block',
                    'type': 'python_return'
                },
                {
                    'kind': 'block',
                    'type': 'nn_compact'
                }


            ]
        },
        {
            'kind': 'category',
            'name': 'Loops',
            'categorystyle': 'loop_category',
            'contents': [
                {
                    "kind": "block",
                    "type": "controls_repeat_ext",
                    "inputs": {
                        "TIMES": {
                            'shadow': {
                                'type': 'string',
                                'fields': {
                                    'NUM': 1,
                                },
                            },

                        }
                    }
                },
                {
                    'kind': 'block',
                    'type': 'controls_whileUntil',
                },
                {
                    'kind': 'block',
                    'type': 'controls_for',
                    'inputs': {
                        'FROM': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 1,
                                },
                            },
                        },
                        'TO': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 10,
                                },
                            },
                        },
                        'BY': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 1,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'controls_forEach',
                },
                {
                    'kind': 'block',
                    'type': 'controls_flow_statements',
                },
            ],
        },
        {
            'kind': 'category',
            'name': 'Math',
            'categorystyle': 'math_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'math_number',
                    'fields': {
                        'NUM': 123,
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_arithmetic',
                    'inputs': {
                        'A': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 1,
                                },
                            },
                        },
                        'B': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 1,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_single',
                    'inputs': {
                        'NUM': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 9,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_trig',
                    'inputs': {
                        'NUM': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 45,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_constant',
                },
                {
                    'kind': 'block',
                    'type': 'math_number_property',
                    'inputs': {
                        'NUMBER_TO_CHECK': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 0,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_round',
                    'fields': {
                        'OP': 'ROUND',
                    },
                    'inputs': {
                        'NUM': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 3.1,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_on_list',
                    'fields': {
                        'OP': 'SUM',
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_modulo',
                    'inputs': {
                        'DIVIDEND': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 64,
                                },
                            },
                        },
                        'DIVISOR': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 10,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_constrain',
                    'inputs': {
                        'VALUE': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 50,
                                },
                            },
                        },
                        'LOW': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 1,
                                },
                            },
                        },
                        'HIGH': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 100,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_random_int',
                    'inputs': {
                        'FROM': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 1,
                                },
                            },
                        },
                        'TO': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 100,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'math_random_float',
                },
                {
                    'kind': 'block',
                    'type': 'math_atan2',
                    'inputs': {
                        'X': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 1,
                                },
                            },
                        },
                        'Y': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 1,
                                },
                            },
                        },
                    },
                },
            ],
        },
        {
            'kind': 'category',
            'name': 'JAX/Flax Operation',
            'categorystyle': 'procedure_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'Add_vectors',
                    'inputs': {
                        "Array1": {
                            "type": "Array",
                            "message0": "Array1 %1",
                            "args0": [
                                {
                                    "type": "input_dummy"
                                }
                            ],
                            "inputsInline": true,
                            "previousStatement": null,
                            "nextStatement": null,
                            "colour": 230
                        },
                        "Array2": {
                            "type": "Array",
                            "message0": "Array2 %1",
                            "args0": [
                                {
                                    "type": "input_dummy"
                                }
                            ],
                            "inputsInline": true,
                            "previousStatement": null,
                            "nextStatement": null,
                            "colour": 230
                        }
                    },
                },
                {
                    'kind': 'block',
                    'type': 'self',
                    'inputs': {
                        "func": {
                            'shadow': {
                                "type": 'math_number',
                                'fields': {
                                    'NUM': '',
                                }
                            },
                            "inputsInline": true,
                            "previousStatement": null,
                            "nextStatement": null,
                            "colour": 230
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'generate_randomePRNG',
                    'inputs': {
                        "seed": {
                            "type": "text",
                            "message0": "Seed %1",
                            "args0": [
                                {
                                    "type": "input_dummy"
                                }
                            ],
                            "inputsInline": true,
                            "previousStatement": null,
                            "nextStatement": null,
                            "colour": 230
                        },
                    },
                },
                {'kind': 'block', 'type': 'reshape'}, // Newly added block
                {'kind': 'block', 'type': 'Dense'},
                {'kind': 'block', 'type': 'max_pool_layer'},
                {'kind': 'block', 'type': 'relu'},
                //{'kind': 'block', 'type': 'dropout_layer'},
                {'kind': 'block', 'type': 'tanh_layer'},
                {'kind': 'block', 'type': 'sigmoid_layer'},
                {'kind': 'block', 'type': 'rnn_layer'},
                {'kind': 'block', 'type': 'gelu'},

            ],
        },
        {
            'kind': 'category',
            'name': 'Activation Function',
            'categorystyle': 'text_category',
            'contents':[



                {'kind':'block','type':'comment'},
            ],},
        {
            'kind': 'category',
            'name': 'Layers',
            'colour': 120,
            'contents': [
                {
                    'kind': 'category',
                    'name': 'Linear Modules',
                    'colour': 230,
                    'contents': [
                        {'kind': 'block', 'type': 'Dense'},
                        //{'kind': 'block', 'type': 'dense_general'},
                        {'kind': 'block', 'type': 'reshape'},
                        {'kind': 'block', 'type': 'Conv'},
                        {'kind': 'block', 'type': 'convTranspose'},
                        {'kind': 'block', 'type': 'conv_local'},
                        {'kind': 'block', 'type': 'embed'},
                        {'kind': 'block', 'type': 'einsum'},

                    ],
                },
                {
                    'kind': 'category',
                    'name': 'Pooling',
                    'colour': 130,
                    'contents': [
                        {'kind': 'block', 'type': 'max_pool_layer'},
                        {'kind': 'block', 'type': 'avg_pool'},
                        {'kind': 'block', 'type': 'pool_layer'},


                    ],
                },
                {
                    'kind': 'category',
                    'name': 'Normalisation',
                    'colour': 50,
                    'contents': [
                        {'kind': 'block', 'type': 'batch_norm_layer'},
                        {'kind': 'block', 'type': 'layer_norm_layer'},
                        {'kind': 'block', 'type': 'group_norm_layer'},
                        {'kind': 'block', 'type': 'RMS_norm_layer'},
                        {'kind': 'block', 'type': 'Spectral_norm_layer'},
                        {'kind': 'block', 'type': 'Instance_norm_layer'},
                        {'kind': 'block', 'type': 'Weight_norm_layer'},


                    ],
                },
                {
                    'kind': 'category',
                    'name': 'Combinators',
                    'colour': 300,
                    'contents': [
                        {'kind': 'block', 'type': 'Sequential'}
                    ],
                },
                {
                    'kind': 'category',
                    'name': 'Stochastic',
                    'colour': 180,
                    'contents': [
                        {'kind': 'block', 'type': 'Dropout'},
                    ],
                },
                {
                    'kind': 'category',
                    'name': 'Attention',
                    'colour': 70,
                    'contents': [
                        {'kind': 'block', 'type': 'MultiHeadDotProductAttention'},
                        {'kind': 'block', 'type': 'MultiHeadAttention'},
                        {'kind': 'block', 'type': 'SelfAttention'},
                        {'kind': 'block', 'type': 'DotProductAttentionWeights'},
                        {'kind': 'block', 'type': 'DotProductAttention'},
                        {'kind': 'block', 'type': 'makeattentionmask'},
                        {'kind': 'block', 'type': 'makecausalmask'},
                    ],
                },
                {
                    'kind': 'category',
                    'name': 'Recurrent',
                    'colour': 100,
                    'contents': [
                        {'kind': 'block', 'type': 'RNNCellBase'},
                        {'kind': 'block', 'type': 'LSTMCell'},
                        {'kind': 'block', 'type': 'OptimizedLSTMCell'},
                        {'kind': 'block', 'type': 'ConvLSTMCell'},
                        {'kind': 'block', 'type': 'SimpleCell'},
                        {'kind': 'block', 'type': 'GRUCell'},
                        {'kind': 'block', 'type': 'MGUCell'},
                        {'kind': 'block', 'type': 'rnn_layer'},
                        {'kind': 'block', 'type': 'Bidirectional'},
                    ],
                },
            ],
        },
        {
            'kind': 'category',
            'name': 'Activation Functions',
            'colour':230,
            'contents': [
                {'kind': 'block', 'type': 'PReLU'},
                {
                    'kind': 'block',
                    'type': 'celu_layer'
                },
                {
                    'kind': 'block',
                    'type': 'elu_layer'
                },
                {
                    'kind': 'block',
                    'type': 'gelu_layer'
                },
                {
                    'kind': 'block',
                    'type': 'glu_layer'
                },
                {
                    'kind': 'block',
                    'type': 'hard_sigmoid_layer'
                },
                {
                    'kind': 'block',
                    'type': 'hard_silu_layer'
                },
                {'kind': 'block', 'type': 'sigmoid_layer'},
                {'kind': 'block', 'type': 'tanh_layer'},
                {'kind': 'block', 'type': 'relu_layer'},
                {'kind': 'block', 'type': 'hard_tanh'},
                {'kind': 'block', 'type': 'leaky_relu'},
                {'kind': 'block', 'type': 'log_sigmoid'},
                {'kind': 'block', 'type': 'log_softmax'},
                {'kind': 'block', 'type': 'logsumexp'},
                {'kind': 'block', 'type': 'one_hot'},
                {'kind': 'block', 'type': 'selu'},
                {'kind': 'block', 'type': 'sigmoid_layer'},
                {'kind': 'block', 'type': 'silu'},
                {'kind': 'block', 'type': 'soft_sign'},
                {'kind': 'block', 'type': 'softmax'},
                {'kind': 'block', 'type': 'softplus'},
                {'kind': 'block', 'type': 'standardize'},
                {'kind': 'block', 'type': 'swish'},


                // Add more activation functions as needed
            ],
        },
        // Other blocks in the flax.linen package can go here
        {
            'kind': 'category',
            'name': 'Module',
            'contents': [

            ],
        },
        {
            'kind': 'category',
            'name': 'Init/Apply',
            'contents': [

            ],
        },
        {
            'kind': 'category',
            'name': 'Initializers',
            'contents': [

            ],
        },
        {
            'kind': 'category',
            'name': 'Transformation',
            'contents': [

            ],
        },
        {
            'kind': 'category',
            'name': 'Inspection',
            'contents': [

            ],
        },
        {
            'kind': 'category',
            'name': 'Variable Dictionary',
            'contents': [

            ],
        },
        {
            'kind': 'category',
            'name': 'SPMD',
            'contents': [

            ],
        },
        {
            'kind': 'category',
            'name': 'Decorators',
            'contents': [

            ],
        },
        {
            'kind': 'category',
            'name': 'Profiling',
            'contents': [

            ],
        },
        // Add other blocks outside the subgroups as needed
        {
            'kind': 'category',
            'name': 'Text',
            'categorystyle': 'text_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'text',
                },
                {
                    'kind': 'block',
                    'type': 'text_multiline',
                },
                {
                    'kind': 'block',
                    'type': 'text_join',
                },
                {
                    'kind': 'block',
                    'type': 'text_append',
                    'inputs': {
                        'TEXT': {
                            'shadow': {
                                'type': 'text',
                                'fields': {
                                    'TEXT': '',
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_length',
                    'inputs': {
                        'VALUE': {
                            'shadow': {
                                'type': 'text',
                                'fields': {
                                    'TEXT': 'abc',
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_isEmpty',
                    'inputs': {
                        'VALUE': {
                            'shadow': {
                                'type': 'text',
                                'fields': {
                                    'TEXT': '',
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_indexOf',
                    'inputs': {
                        'VALUE': {
                            'block': {
                                'type': 'variables_get',
                            },
                        },
                        'FIND': {
                            'shadow': {
                                'type': 'text',
                                'fields': {
                                    'TEXT': 'abc',
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_charAt',
                    'inputs': {
                        'VALUE': {
                            'block': {
                                'type': 'variables_get',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_getSubstring',
                    'inputs': {
                        'STRING': {
                            'block': {
                                'type': 'variables_get',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_changeCase',
                    'inputs': {
                        'TEXT': {
                            'shadow': {
                                'type': 'text',
                                'fields': {
                                    'TEXT': 'abc',
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_trim',
                    'inputs': {
                        'TEXT': {
                            'shadow': {
                                'type': 'text',
                                'fields': {
                                    'TEXT': 'abc',
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_count',
                    'inputs': {
                        'SUB': {
                            'shadow': {
                                'type': 'text',
                            },
                        },
                        'TEXT': {
                            'shadow': {
                                'type': 'text',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_replace',
                    'inputs': {
                        'FROM': {
                            'shadow': {
                                'type': 'text',
                            },
                        },
                        'TO': {
                            'shadow': {
                                'type': 'text',
                            },
                        },
                        'TEXT': {
                            'shadow': {
                                'type': 'text',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'text_reverse',
                    'inputs': {
                        'TEXT': {
                            'shadow': {
                                'type': 'text',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'add_text',
                    'inputs': {
                        'TEXT': {
                            'shadow': {
                                'type': 'text',
                                'fields': {
                                    'TEXT': 'abc',
                                },
                            },
                        },
                        'COLOR': {
                            'shadow': {
                                'type': 'colour_picker',
                                'fields': {
                                    'COLOUR': '#aa00cc',
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'DataWrapper',
                    'inputs': {
                        'TEXT': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'TEXT': '0',
                                },
                            },
                        },
                    },
                },
            ],
        },
        {
            'kind': 'category',
            'name': 'Lists',
            'categorystyle': 'list_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'lists_create_with',
                },
                {
                    'kind': 'block',
                    'type': 'lists_create_with',
                },
                {
                    'kind': 'block',
                    'type': 'lists_repeat',
                    'inputs': {
                        'NUM': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 5,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'lists_length',
                },
                {
                    'kind': 'block',
                    'type': 'lists_isEmpty',
                },
                {
                    'kind': 'block',
                    'type': 'lists_indexOf',
                    'inputs': {
                        'VALUE': {
                            'block': {
                                'type': 'variables_get',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'lists_getIndex',
                    'inputs': {
                        'VALUE': {
                            'block': {
                                'type': 'variables_get',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'lists_setIndex',
                    'inputs': {
                        'LIST': {
                            'block': {
                                'type': 'variables_get',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'lists_getSublist',
                    'inputs': {
                        'LIST': {
                            'block': {
                                'type': 'variables_get',
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'lists_split',
                    'inputs': {
                        'DELIM': {
                            'shadow': {
                                'type': 'text',
                                'fields': {
                                    'TEXT': ',',
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'lists_sort',
                },
                {
                    'kind': 'block',
                    'type': 'lists_reverse',
                },
            ],
        },
        {
            'kind': 'category',
            'name': 'Variable',
            'categorystyle': 'variables_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'set_var'
                },
                {
                    'kind': 'block',
                    'type': 'get_variable'
                },

            ]
        },
        {
            'kind': 'category',
            'name': 'Color',
            'categorystyle': 'colour_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'colour_picker',
                },
                {
                    'kind': 'block',
                    'type': 'colour_random',
                },
                {
                    'kind': 'block',
                    'type': 'colour_rgb',
                    'inputs': {
                        'RED': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 100,
                                },
                            },
                        },
                        'GREEN': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 50,
                                },
                            },
                        },
                        'BLUE': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 0,
                                },
                            },
                        },
                    },
                },
                {
                    'kind': 'block',
                    'type': 'colour_blend',
                    'inputs': {
                        'COLOUR1': {
                            'shadow': {
                                'type': 'colour_picker',
                                'fields': {
                                    'COLOUR': '#ff0000',
                                },
                            },
                        },
                        'COLOUR2': {
                            'shadow': {
                                'type': 'colour_picker',
                                'fields': {
                                    'COLOUR': '#3333ff',
                                },
                            },
                        },
                        'RATIO': {
                            'shadow': {
                                'type': 'math_number',
                                'fields': {
                                    'NUM': 0.5,
                                },
                            },
                        },
                    },
                },
            ],
        },
        {
            'kind': 'sep',
        },
        {
            'kind': 'category',
            'name': 'Text',
            'categorystyle': 'text_category',
            'contents': [
                {
                    'kind': 'block',
                    'type': 'string'
                },

            ]
        },
        {
            'kind': 'category',
            'name': 'Functions',
            'categorystyle': 'procedure_category',
            'custom': 'PROCEDURE',
        },

    ],

};







































