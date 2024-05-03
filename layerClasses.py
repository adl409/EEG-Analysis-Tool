class Layer:
    def __init__(self) -> None:
        pass

class Dense(Layer):
    
    def __init__(self, index, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'):
        
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.layerType = "dense"
        self.displayName = "Dense"

class Flatten(Layer):
    
    def __init__(self, index):
        self.layerType = "flatten"    
        
        self.displayName = "Flatten"

class Zero_Padding_2d(Layer):
    
    def __init__(self,index, padding=(1,1)):
        self.index= index
        self.padding= padding
        self.layerType = "zero_padding_2d"
        self.displayName = "Zero Padding 2D"

class Average_Pooling_2d(Layer):
    
    def __init__(self, index, pool_size=(1,1), strides=None,padding='valid'):
        
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        self.layerType = "average_pooling_2d"
        self.displayName = "Average Pooling 2D"

class Max_Pool_2d(Layer):
    
    def __init__(self,index,pool_size=(2,2), strides=None,padding='valid'):
        
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        self.layerType = "max_pool_2d"
        self.displayName = "Max Pooling 2D"
    
class Convolution_2d(Layer):
    
    def __init__(self,index,filters=None,kernel_size=None,strides=(1,1),padding='valid', dialation_rate=(1,1),
                 groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'):
        
        self.filter=filters
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
        self.displayName = "Convolutional 2D"

class Convolution_2d_Transpose(Layer):
    
    def __init__(self,index,filters=None,kernel_size=None,strides=(1,1),padding='valid', dialation_rate=(1,1),
                 groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'):
        
        self.filter=filters
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
        self.displayName = "Transposed Convolutional 2D"

class Depthwise_Conv_2d(Layer):
     
     def __init__(self,index,kernel_size=None,strides=(1,1),padding='valid', dialation_rate=(1,1),
                  depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform',bias_initializer='zeros'):
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
        self.displayName = "Depthwise Convolutional 2D"

class Separable_Conv_2d(Layer):
     
     def __init__(self,index,filters=None,kernel_size=None,strides=(1,1),padding='valid', 
                  dialation_rate=(1,1),depth_multiplier=1, activation=None, use_bias=True, 
                  depthwise_initializer='glorot_uniform',pointwise_initializer='glorot_uniform',bias_initializer='zeros'):
        
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
        self.displayName = "Separable Convolutional 2D"

class Conv_LSTM_2d(Layer):
    
   def __init__(self, index, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', 
                 data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', dropout=0.0, recurrent_dropout=0.0, seed=None):
        
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
        self.displayName = "Convolutional LSTM 2D"

class SimpleRNN(Layer):
    
    def __init__(self, index, units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros',
                 dropout=0.0, recurrent_dropout=0.0, seed=None):
        
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
        self.displayName = "Recurrent"

class LSTM(Layer):
    
   def __init__(self, index, units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', unit_forget_bias=True, dropout=0.0, recurrent_dropout=0.0,seed=None):
        
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
        self.displayName = "LSTM"
                     
class GRU(Layer):
    
    def __init__(self, index, units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', unit_forget_bias=True, dropout=0.0, recurrent_dropout=0.0,seed=None,reset_after=True):
        
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
        self.displayName = "Gated Recurrent Unit"

