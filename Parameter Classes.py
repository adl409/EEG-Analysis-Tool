class Dense:
    def __init__(self, units, index=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'):
        self.index = index
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

class Flatten:
    def __init(self, index=None):
        self.index = index

class Zero_Padding_2d:
    def __init__(self,index=None, padding=(1,1)):
        self.index= index
        self.padding= padding

class Average_Pooling_2d:
    def __init__(self, index=None, pool_size=(1,1), strides=(1,1),padding='valid',dialation_rate=(1,1),groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'):
        self.index=index
        self.pool_size=pool_size
        self.strides=strides
        self.padding=padding
        self.dialation_rate=dialation_rate
        self.groups=groups
        self.kernel_intializer=kernel_initializer
        self.bias_initializer=bias_initializer
