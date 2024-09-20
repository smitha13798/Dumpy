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
pythonGenerator.forBlock['nn_nowrap'] = function(block) {
    return `@nn.nowrap\n`;
};
pythonGenerator.forBlock['cheatblock'] = function(block) {
    var input = block.getFieldValue('VAR')+'\n';
    return input;
};
pythonGenerator.forBlock['set_var'] = function(block) {
    var variable = block.getFieldValue('SET_VARIABLE');
    const value = pythonGenerator.valueToCode(block, 'VALUE', Order.NONE);
    return `${variable} = ${value}\n`;
    //return [`${variable} = ${value}`,Order.ATOMIC]
};

pythonGenerator.forBlock['member'] = function(block) {
    var variable = block.getFieldValue('SET_VARIABLE');
    const value = pythonGenerator.valueToCode(block, 'VALUE', Order.NONE);
    return `${variable} : ${value}\n`;
    //return [`${variable} = ${value}`,Order.ATOMIC]
};
pythonGenerator.forBlock['Declaration'] = function(block) {
    var variable = block.getFieldValue('SET_VARIABLE');
    return `${variable}\n`;
    //return [`${variable} = ${value}`,Order.ATOMIC]
};

pythonGenerator.forBlock['get_variable'] = function(block) {
    const variableName = block.getFieldValue('VARIABLE_NAME');
    const code = `${variableName}`;
    return [code, Order.ATOMIC];
};

pythonGenerator.forBlock['relu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.relu(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['PReLU'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.PReLU(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['hard_tanh'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.hard_tanh(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['leaky_relu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.leaky_relu(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['log_sigmoid'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.log_sigmoid(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['log_softmax'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.log_softmax(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['soft_sign'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.soft_sign(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['softmax'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.softmax(${variableName})\n`;
    return [code,, Order.ATOMIC];
};
pythonGenerator.forBlock['softplus'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.softplus(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['standardize'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.standardize(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['swish'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.swish(${variableName})\n`;
    return [code, Order.ATOMIC];
};

pythonGenerator.forBlock['logsumexp'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.logsumexp(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['one_hot'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.one_hot(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['selu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.one_hot(${variableName})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['celu_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.celu(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['elu_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.elu(${variableName})`;
    return [code, Order.ATOMIC];
};

pythonGenerator.forBlock['gelu_layer'] = function(block) {
    const code = `nn.gelu(x)\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['glu_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.glu(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['hard_sigmoid_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.hard_sigmoid(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['hard_silu_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.hard_silu(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['Embed'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.Embed(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['einsum'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.Einsum(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['jit'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.jit(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['remat'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.remat(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['scan'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.scan(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['vmap'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.vmap(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['remat_scan'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.remat_scan(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['map_variables'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.map_variables(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['jvp'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.jvp(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['vjp'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.vjp(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['custom_vjp'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.custom_vjp(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['while_loop'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.while_loop(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['cond'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.cond(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['switchblock'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.switch(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['tabulate'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.tabulate(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Variable'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.Variable(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['gelu'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.gelu${variableName}\n`;
    return `nn.gelu${variableName}\n`;

};
pythonGenerator.forBlock['comment'] = function(block, generator) {
    const text = block.getFieldValue( 'VAR');

    return `# ${text}\n`;

};


pythonGenerator.forBlock['reshape'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `${variableName}.reshape((${variableName}.shape[0], -1))\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Dense'] = function(block, generator) {
    const units = block.getFieldValue('UNITS') || '(x)';
    const variableName = block.getFieldValue( 'VAR');
    let code = ""

    code = `nn.Dense(${units})(${variableName})\n`;
    if(variableName===""){
        code = `nn.Dense(${units})\n`;
    }
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Sequential'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.Sequential(${units})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['max_pool_layer'] = function(block, generator) {
    const windowShapeX = block.getFieldValue('WINDOW_SHAPE_X');
    const windowShapeY = block.getFieldValue('WINDOW_SHAPE_Y');
    const strideX = block.getFieldValue('STRIDE_X');
    const strideY = block.getFieldValue('STRIDE_Y');
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.max_pool(window_shape=(${windowShapeX}, ${windowShapeY}), strides=(${strideX}, ${strideY}))(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['pool_layer'] = function(block, generator) {
    const windowShapeX = block.getFieldValue('WINDOW_SHAPE_X');
    const windowShapeY = block.getFieldValue('WINDOW_SHAPE_Y');
    const strideX = block.getFieldValue('STRIDE_X');
    const strideY = block.getFieldValue('STRIDE_Y');
    const variableName = block.getFieldValue( 'VAR');
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
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.Dropout(rate=${rate})(${variableName})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['batch_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.BatchNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['layer_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.LayerNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['group_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.GroupNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['RMS_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.RMSNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Instance_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.InstanceNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Spectral_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.SpectralNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Weight_norm_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.WeightNorm()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['MultiHeadAttention'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.MultiHeadAttention()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['MultiHeadDotProductAttention'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.MultiHeadDotProductAttention()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['SelfAttention'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.SelfAttention()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['DotProductAttention'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.dot_product_attention()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['DotProductAttentionWeights'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.dot_product_attention_weights()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['makeattentionmask'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.make_attention_mask()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['makecausalnmask'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.make_causal_mask()(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['tanh_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.tanh(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['sigmoid_layer'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.sigmoid(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['rnn_layer'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.RNN(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['RNNCellBase'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.RNNCellBase(${units})\n`;
    return [code, Order.ATOMIC];
};
pythonGenerator.forBlock['LSTMCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.LSTMCell(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['OptimizedLSTMCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.OptimizedLSTMCell(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['ConvLSTMCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.ConvLSTMCell(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['SimpleCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.SimpleCell(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['GRUCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.GRUCell(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['MGUCell'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.MGUCell(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Bidirectional'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `nn.Bidirectional(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Dense_General'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.DenseGeneral(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};


pythonGenerator.forBlock['convTranspose'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') ;
    let code = `nn.ConvTranspose(${params})(${variableName})\n`;

    if(variableName===""){
        code = `nn.ConvTranspose(${params})\n`;
    }
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['conv_local'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.ConvLocal(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Conv'] = function(block, generator) {
    const params = block.getFieldValue('PARAMS');
    const variableName = block.getFieldValue( 'VAR') ;
    const code = `nn.Conv(${params})(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['constant'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.constant(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['delta_orthogonal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.delta_orthogonal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['glorot_normal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.glorot_normal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['glorot_uniform'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.glorot_uniform(${units})\n`;
    return [code, Order.ATOMIC];

};

pythonGenerator.forBlock['he_normal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.he_normal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['he_uniform'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.he_uniform(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['kaiming_normal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.kaiming_normal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['lecun_normal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.lecun_normal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['lecun_uniform'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.lecun_uniform(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['normal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.normal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['kaiming_uniform'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.kaiming_uniform(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['truncated_normal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.truncated_normal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['ones'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.ones(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['orthogonal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.orthogonal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['xavier_uniform'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.xavier_uniform(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['xavier_normal'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.xavier_normal(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['ones_init'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.ones_init(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['uniform'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.uniform(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['variance_scaling'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.variance_scaling(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['zeros_init'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.zeros_init(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['zeros'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.zeros(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['apply'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.apply(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['init'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.initializers.init(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['init_with_output'] = function(block, generator) {
    const units = block.getFieldValue('UNITS')|| "";
    const code = `jax.nn.init_with_outputializers.init_with_output(${units})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['Partitioned'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.Partitioned(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['get_partition_spec'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.get_partition_spec(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['get_sharding'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.get_sharding(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['LogicallyPartitioned'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.LogicallyPartitioned(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['logical_axis_rules'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.logical_axis_rules(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['logical_to_mesh_axes'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.logical_to_mesh_axes(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['set_logical_axis_rules'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.set_logical_axis_rules(${variableName})\n`;
    return [code, Order.ATOMIC];

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
pythonGenerator.forBlock['string'] = function(block, generator) {

    const text = block.getFieldValue( 'VAR');

    return [text, Order.ATOMIC];

};

pythonGenerator.forBlock['get_logical_axis_rules'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.get_logical_axis_rules(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['logical_to_mesh_sharding'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.logical_to_mesh_sharding(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['with_logical_constraint'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.with_logical_constraint(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['with_logical_partitioning'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.with_logical_partitioning(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['with_partitioning'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.with_partitioning(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['logical_to_mesh'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.logical_to_mesh(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['enable_named_call'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.enable_named_call(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['disable_named_call'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.disable_named_call(${variableName})\n`;
    return [code, Order.ATOMIC];

};
pythonGenerator.forBlock['override_named_call'] = function(block, generator) {
    const variableName = block.getFieldValue( 'VAR');
    const code = `nn.override_named_call(${variableName})\n`;
    return [code, Order.ATOMIC];

};

















