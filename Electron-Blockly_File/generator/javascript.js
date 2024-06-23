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


pythonGenerator.forBlock['python_function'] = function(block) {
    const functionName = block.getFieldValue('CLASS_NAME');
    let methods = pythonGenerator.statementToCode(block, 'METHODS');
    if (methods) {
        methods = pythonGenerator.prefixLines(methods, pythonGenerator.INDENT);
    }
    return `@nn.compact\ndef ${functionName}:\n${methods}`;
};

pythonGenerator.forBlock['self'] = function(block, generator) {
    var model = generator.valueToCode(block, 'func', Order.NONE);
    // TODO: Assemble python into code variable.
    // TODO: Change ORDER_NONE to the correct strength.
    return '@nn.compact\n' +
        '    def __call__(self, x):\n';
};

pythonGenerator.forBlock['conv_layer'] = function(block) {
    const features = block.getFieldValue('FEATURES');
    const kernelSizeX = block.getFieldValue('KERNEL_SIZE_X');
    const kernelSizeY = block.getFieldValue('KERNEL_SIZE_Y');
    return `x = nn.Conv(features=${features}, kernel_size=(${kernelSizeX}, ${kernelSizeY}))(x)\n`;
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
pythonGenerator.forBlock['relu_layer'] = function(block) {
    return `x = nn.relu(x)\n`;
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
pythonGenerator.forBlock['average_pool_layer'] = function(block) {
    const poolSizeX = block.getFieldValue('POOL_SIZE_X');
    const poolSizeY = block.getFieldValue('POOL_SIZE_Y');
    const strideX = block.getFieldValue('STRIDE_X');
    const strideY = block.getFieldValue('STRIDE_Y');
    return `x = nn.AvgPool(window_shape=(${poolSizeX}, ${poolSizeY}), strides=(${strideX}, ${strideY}))(x)\n`;
};

pythonGenerator.forBlock['dropout_layer'] = function(block) {
    const rate = block.getFieldValue('RATE');
    return `x = nn.Dropout(rate=${rate})(x)\n`;
};

pythonGenerator.forBlock['batch_norm_layer'] = function(block) {
    return `x = nn.BatchNorm()(x)\n`;
};
pythonGenerator.forBlock['tanh_layer'] = function(block) {
    return `x = nn.tanh(x)\n`;
};

pythonGenerator.forBlock['sigmoid_layer'] = function(block) {
    return `x = nn.sigmoid(x)\n`;
};

pythonGenerator.forBlock['rnn_layer'] = function(block) {
    const units = block.getFieldValue('UNITS');
    const returnSeq = block.getFieldValue('RETURN_SEQ') === 'TRUE' ? 'True' : 'False';
    return `x = nn.RNN(units=${units}, return_sequences=${returnSeq})(x)\n`;
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








    













