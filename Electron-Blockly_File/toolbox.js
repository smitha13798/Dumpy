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
          'kind': 'block',
          'type': 'controls_repeat_ext',
          'inputs': {
            'TIMES': {
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
        {'kind': 'block', 'type': 'flatten_layer'}, // Newly added block
        {'kind': 'block', 'type': 'dense_layer'},
        {'kind': 'block', 'type': 'max_pool_layer'},
        {'kind': 'block', 'type': 'relu_layer'},
        {'kind': 'block', 'type': 'average_pool_layer'},
        {'kind': 'block', 'type': 'dropout_layer'},
        {'kind': 'block', 'type': 'batch_norm_layer'},
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
      ],},
    {
      'kind': 'category',
      'name': 'flax.linen package',
      'categorystyle': 'text_category',
      'contents':[
        {'kind': 'block', 'type': 'embed'},
        {'kind': 'block', 'type': 'dense_general'},
        {'kind': 'block', 'type': 'tabulate'},
        {'kind': 'block', 'type': 'vmap'},
        {'kind': 'block', 'type': 'scan'},
        {'kind': 'block', 'type': 'jit'},
        {'kind': 'block', 'type': 'remat'},
        {'kind': 'block', 'type': 'convTranspose'},
        {'kind': 'block', 'type': 'conv_local'},
        {'kind': 'block', 'type': 'conv_layer'},
      ],},
    {
      'kind': 'category',
      'name': 'Data Handling',
      'categorystyle': 'text_category',
      'contents': [
        {
          'kind': 'block',
          'type': 'dataset_selection'
        },
        {
          'kind': 'block',
          'type': 'data_loader_config'
        },
        {
          'kind': 'block',
          'type': 'data_preprocessing'
        },
        {
          'kind': 'block',
          'type': 'data_batching'
        },
        {
          'kind': 'block',
          'type': 'data_shuffling'
        },
        {
          'kind': 'block',
          'type': 'data_transformations'
        },
        {
          'kind': 'block',
          'type': 'split_data'
        }
      ]
    },
    {
      'kind': 'category',
      'name': 'Training and Evaluation',
      'categorystyle': 'procedure_category',
      'contents': [
        {'kind': 'block', 'type': 'loss_function'},
        {'kind': 'block', 'type': 'optimizer'},
        {'kind': 'block', 'type': 'training_step'},
        {'kind': 'block', 'type': 'evaluation'},
        {'kind': 'block', 'type': 'training_loop'}
      ]
    },
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
      'name': 'Functions',
      'categorystyle': 'procedure_category',
      'custom': 'PROCEDURE',
    },

  ],

};
