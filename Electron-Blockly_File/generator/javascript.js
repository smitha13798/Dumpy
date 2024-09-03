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
    const value = pythonGenerator.valueToCode(block, 'VALUE', Order.NONE)||"''";
    return `${variable} = ${value}\n`;
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




















