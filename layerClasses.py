class Dense:
    
    def __init__(self, index, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'):
        self.index = index
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.layerType = "dense"

class Flatten:
    
    def __init__(self, index):
        self.layerType = "flatten"    
        self.index = index

class Zero_Padding_2d:
    
    def __init__(self,index, padding=(1,1)):
        self.index= index
        self.padding= padding
        self.layerType = "zero_padding_2d"

class Average_Pooling_2d:
    
    def __init__(self, index, pool_size=(1,1), strides=None,padding='valid'):
        self.index=index
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        self.layerType = "average_pooling_2d"

class Max_Pool_2d:
    
    def __init__(self,index,pool_size=(2,2), strides=None,padding='valid'):
        self.index=index
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        self.layerType = "max_pool_2d"
    
class Convolution_2d:
    
    def __init__(self,index,filters=None,kernel_size=None,strides=(1,1),padding='valid', dialation_rate=(1,1),
                 groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'):
        self.index=index
        self.filter=filter
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.dialation_rate=dialation_rate
        self.groups=groups
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.layerType = "convolution_2d"

class Convolution_2d_Transpose:
    
    def __init__(self,index,filters=None,kernel_size=None,strides=(1,1),padding='valid', dialation_rate=(1,1),
                 groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'):
        self.index=index
        self.filter=filter
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.dialation_rate=dialation_rate
        self.groups=groups
        self.activation=activation
        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.layerType = "convolution_2d_transpose"

class Depthwise_Conv_2d:
     
     def __init__(self,index,kernel_size=None,strides=(1,1),padding='valid', dialation_rate=(1,1),
                  depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform',bias_initializer='zeros'):
        self.index=index
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.dialation_rate=dialation_rate
        self.depth_multiplier=depth_multiplier
        self.activation=activation
        self.use_bias=use_bias
        self.depthwise_initializer=depthwise_initializer
        self.bias_initializer=bias_initializer
        self.layerType = "depthwise_conv_2d"

class Separable_Conv_2d:
     
     def __init__(self,index,filters=None,kernel_size=None,strides=(1,1),padding='valid', 
                  dialation_rate=(1,1),depth_multiplier=1, activation=None, use_bias=True, 
                  depthwise_initializer='glorot_uniform',pointwise_initializer='glorot_uniform',bias_initializer='zeros'):
        self.index=index
        self.filter=filter
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.dialation_rate=dialation_rate
        self.depth_multiplier=depth_multiplier
        self.activation=activation
        self.use_bias=use_bias
        self.depthwise_initializer=depthwise_initializer
        self.pointwise_initializer=pointwise_initializer
        self.bias_initializer=bias_initializer
        self.layerType = "separable_conv_2d"

class Conv_LSTM_2d:
    
   def __init__(self, index, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', 
                 data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', dropout=0.0, recurrent_dropout=0.0, seed=None):
        self.index = index
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.seed=seed
        self.layerType = "conv_lstm_2d"

class SimpleRNN:
    
    def __init__(self, index, units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros',
                 dropout=0.0, recurrent_dropout=0.0, seed=None):
        self.index = index
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.seed=seed
        self.layerType = "simplernn"

class LSTM:
    
   def __init__(self, index, units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', unit_forget_bias=True, dropout=0.0, recurrent_dropout=0.0,seed=None):
        self.index = index
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.seed=seed
        self.layerType = "lstm"
                     
class GRU:
    
    def __init__(self, index, units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', unit_forget_bias=True, dropout=0.0, recurrent_dropout=0.0,seed=None,reset_after=True):
        self.index = index
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = unit_forget_bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.seed=seed
        self.reset_after=reset_after
        self.layerType = "gru"

