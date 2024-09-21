from enum import Enum

#CAP TEXT IS FUNCTION NAME AND ="is type from toolbox
# Class-based enum definition
class FunctionType(Enum):
    CELU_LAYER = "celu_layer"
    ELU_LAYER = "elu_layer"
    GELU_LAYER = "gelu_layer"
    GLU_LAYER = "glu_layer"
    HARD_SIGMOID_LAYER = "hard_sigmoid_layer"
    HARD_SILU_LAYER = "hard_silu_layer"
    CONVTRANSPOSE = "ConvTranspose"
    GELU = "gelu"
    PYTHON_FUNCTION = "python_function"
    PYTHON_CLASS = "python_class"
    DATA_WRAPPER_G = "DataWrapperG"
    ADD_TEXT = "addText"
    ADD_VECTORS = "AddVectors"
    GENERATE_RANDOM = "generateRandome"
    RESHAPE = "reshape"
    DENSE = "Dense"
    MAX_POOL_LAYER = "maxPoolLayer"
    RELU = "relu"
    CONV = "Conv"
    SELF = "self"
    BATCH_NORM = "batchNorm"
    AVG_POOL = "avg_pool"
    DROPOUT = "dropout"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RNN = "rnn"
    DATA_BATCHING_BLOCK = "dataBatchingBlock"
    DATA_LOADER_BLOCK = "dataLoaderBlock"
    DATA_SELECTION_BLOCK = "dataselectionBlock"
    DATA_PREPROCESSING_BLOCK = "dataPreprocessingBlock"
    DATA_SHUFFLING_BLOCK = "dataShufflingBlock"
    TRANSFORMATIONS_BLOCK = "transformationsBlock"
    SPLIT_DATA_BLOCK = "splitDataBlock"
    LOSS_FUNCTION_BLOCK = "lossFunctionBlock"
    OPTIMIZER_BLOCK = "optimizerBlock"
    TRAINING_STEP_BLOCK = "trainingStepBlock"
    TRAINING_LOOP_BLOCK = "trainingLoopBlock"
    EVALUATION_BLOCK = "evaluationBlock"
    PYTHON_CLASS_ATTRIBUTE = "python_class_attribute"
    PYTHON_RETURN = "python_return"
    NN_COMPACT = "nn_compact"
    NN_NOWRAP = "nn_nowrap"
    SET_VARIABLE_BLOCK = "setVariableBlock"
    GET_VARIABLE_BLOCK = "getVariableBlock"
    RANGE = "range"
    CONTROLS_REPEAT_EXT = "controls_repeat_ext"
    SELFATTENTION = "SelfAttention"
    VARIABLE = "variable"
    DECLARATION = "Declaration"
    # Newly added types
    DELTA_ORTHOGONAL = "delta_orthogonal"
    GLOROT_NORMAL = "glorot_normal"
    GLOROT_UNIFORM = "glorot_uniform"
    HE_NORMAL = "he_normal"
    HE_UNIFORM = "he_uniform"
    EMBED = "embed"
    SCAN = "scan"
    VMAP = "vmap"
    TABULATE = "tabulate"
    KAIMING_NORMAL = "kaiming_normal"
    LECUN_NORMAL = "lecun_normal"
    LECUN_UNIFORM = "lecun_uniform"
    NORMAL_INIT = "normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    TRUNCATED_NORMAL = "truncated_normal"
    ONES_INIT = "ones_init"
    ORTHOGONAL = "orthogonal"
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    UNIFORM_INIT = "uniform"
    VARIANCE_SCALING = "variance_scaling"
    ENABLE_NAMED_CALL = "enable_named_call"
    DISABLE_NAMED_CALL = "disable_named_call"
    ONES = "ones"
    ZEROS_INIT = "zeros_init"
    ZEROS = "zeros"
    SWISH = "swish"
    SELU = "selu"
    SILU = "silu"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    STANDARDIZE = "standardize"
    LOGSUMEXP = "logsumexp"
    ONE_HOT = "one_hot"
    SOFT_SIGN = "soft_sign"
    APPLY = "apply"
    INIT_WITH_OUTPUT = "init_with_output"
    OVERRIDE_NAMED_CALL = "override_named_call"
    DENSE_GENERAL = "Dense_General"
    JIT = "jit"
    REMAT = "remat"
    EINSUM = "einsum"
    POOL_LAYER = "pool_layer"
    MULTIHEAD_ATTENTION = "MultiHeadAttention"
    MULTIHEAD_DOT_PRODUCT_ATTENTION = "MultiHeadDotProductAttention"
    DOT_PRODUCT_ATTENTION = "DotProductAttention"
    DOT_PRODUCT_ATTENTION_WEIGHTS = "DotProductAttentionWeights"
    MAKE_CAUSAL_MASK = "makecausalmask"
    MAKE_ATTENTION_MASK = "makeattentionmask"
    RNN_CELL_BASE = "RNNCellBase"
    LSTM_CELL = "LSTMCell"
    OPTIMIZED_LSTM_CELL = "OptimizedLSTMCell"
    CONV_LSTM_CELL = "ConvLSTMCell"
    SIMPLE_CELL = "SimpleCell"
    GRU_CELL = "GRUCell"
    MGU_CELL = "MGUCell"
    BIDIRECTIONAL = "Bidirectional"
    PRELU = "PReLU"
    HARD_TANH = "hard_tanh"
    LEAKY_RELU = "leaky_relu"
    LOG_SIGMOID = "log_sigmoid"
    LOG_SOFTMAX = "log_softmax"
    REMAT_SCAN = "remat_scan"
    MAP_VARIABLES = "map_variables"
    SWITCH_BLOCK = "switchblock"
    JVP = "jvp"
    CUSTOM_VJP = "custom_vjp"
    WHILE_LOOP = "while_loop"
    COND = "cond"
    PARTITIONED = "Partitioned"
    WITH_PARTITIONING = "with_partitioning"
    GET_PARTITION_SPEC = "get_partition_spec"
    GET_SHARDING = "get_sharding"
    LOGICALLY_PARTITIONED = "LogicallyPartitioned"
    LOGICAL_AXIS_RULES = "logical_axis_rules"
    SET_LOGICAL_AXIS_RULES = "set_logical_axis_rules"
    GET_LOGICAL_AXIS_RULES = "get_logical_axis_rules"
    LOGICAL_TO_MESH_AXES = "logical_to_mesh_axes"
    LOGICAL_TO_MESH = "logical_to_mesh"
    LOGICAL_TO_MESH_SHARDING = "logical_to_mesh_sharding"
    WITH_LOGICAL_CONSTRAINT = "with_logical_constraint"
    WITH_LOGICAL_PARTITIONING = "with_logical_partitioning"
    PYTHON_LOOP = "python_loop"
    STRING = "string"
    MEMBER = "member"

