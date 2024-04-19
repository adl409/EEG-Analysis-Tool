from layerClasses import *
import tensorflow as tf
import numpy as np
from dataParser import *
import time
from sklearn.model_selection import train_test_split

models = {
    "input_parameters": {
        "input_shape1" : 128,
        "input_shape2" : 65,
        #EX dir
        "root_directory" : "~/home/user/dir",
        "normalized" : False
    },
    "test_parameters": {
        "test_split" : 0.20,
        "epochs" : 10,
        #MAYBE? Need more research
        "loss_function": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "shuffle" : True
    },
    "model_1" : {
        "save_model" : True, 
        "save_file" : "mod1",
        "active": True,
        "layers": [
            Flatten(0), Dense(2, 64), Dense(1, 64, activation="relu")
        ]
    },
    "model_2" : {
        "save_model" : False,
        "save_file" : "mod2", 
        "active": False,
        "layers": [
            Max_Pool_2d(1)
        ]
    },  
    "model_3" : {
        "save_model" : False,
        "save_file" : "mod3", 
        "active": False,
        "layers": [
            Dense(1, 64), Flatten(2)
        ]
    },
    "model_4" : {
        "save_model" : False,
        "save_file" : "mod4",
        "active": False,
        "layers": [
            Dense(1, 64), Flatten(2)
        ]
    },
    "model_5" : {
        "save_model" : False,
        "save_file" : "mod5",
        "active": False,
        "layers": [
            Dense(1, 64)
        ]
    }
}

#presetCNN = []

def make_model(modelNum):

    # Pulling layers from dict
    layers = models.get("model_" + str(modelNum)).get("layers")

    # Sorting by index value. Not sure if necessary; depends on if frontend
    # displays layers based on index in array or index value. 
    layers.sort(key=lambda x: x.index)  

    # Creating model
    
    model = tf.keras.Sequential()

    # Configuring input layer
    inputShape1 = models.get("input_parameters").get("input_shape1")
    inputShape2 = models.get("input_parameters").get("input_shape2")
    # Fourth dim added to make compatible with 2d functions
    model.add(tf.keras.layers.Input(shape=(int(inputShape1), int(inputShape2), 1)))

    # Adding layers based on dict
    for layer in layers:
        
        match(layer.layerType):
            case "dense":
                model.add(tf.keras.layers.Dense(layer.units, activation = layer.activation, use_bias = layer.use_bias, kernel_initializer=layer.kernel_initializer, bias_initializer = layer.bias_initializer))
            
            case "flatten": 
                model.add(tf.keras.layers.Flatten())

            case "zero_padding_2d":
                model.add(tf.keras.layers.ZeroPadding2D(padding = layer.padding))

            case "average_pooling_2d":
                model.add(tf.keras.layers.AveragePooling2D(pool_size = layer.pool_size, strides = layer.strides, padding = layer.padding))

            case "max_pool_2d":
                model.add(tf.keras.layers.MaxPooling2D(pool_size=layer.pool_size, strides = layer.strides, padding = layer.padding))

            case "convolution_2d":
                model.add(tf.keras.layers.Conv2d(filters = layer.filter, kernel_size = layer.kernel_size, strides = layer.strides, padding = layer.padding, dialation_rate = layer.dialation_rate, groups = layer.groups, activaiton = layer.activation, use_bias = layer.use_bias, kernel_initializer = layer.kernel_initializer, bias_initializer = layer.bias_initializer))

            case "convolution_2d_transpose":
                model.add(tf.keras.layers.Conv2dTranspose(filters = layer.filter, kernel_size = layer.kernel_size, strides = layer.strides, padding = layer.padding, dialation_rate = layer.dialation_rate, groups = layer.groups, activaiton = layer.activation, use_bias = layer.use_bias, kernel_initializer = layer.kernel_initializer, bias_initializer = layer.bias_initializer))

            case "depthwise_conv_2d":
                model.add(tf.keras.layers.DepthwiseConv2d(kernel_size=layer.kernel_size, strides = layer.strides, padding = layer.padding, dialation_rate = layer.dialation_rate, depth_multiplier = layer.depth_multiplier, activation = layer.activation, use_bias = layer.use_bias, depthwise_initializer = layer.depthwise_initializer, bias_initializer = layer.bias_initializer))

            case "separable_conv_2d":
                print(layer.layerType)

            case "conv_lstm_2d":
                print(layer.layerType)

            case "simplernn":
                print(layer.layerType)

            case "lstm":
                print(layer.layerType)
                
            case "gru":
                print(layer.layerType)

            case _ :
                print("Error Determining layer type...")

    return model

def train_test_model(data, labels, model, modelNum):
     
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels , random_state=104,test_size=models.get("test_parameters").get("test_split"), shuffle=models.get("test_parameters").get("shuffle"))
     
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
     
    model.fit(data_train, labels_train, epochs=models.get("test_parameters").get("epochs"))

    test_loss, test_acc = model.evaluate(data_test,  labels_test, verbose=2)

    if(models.get("model_" + str(modelNum)).get("save_model")):
        model.save(f'./'+ models.get("model_" + str(modelNum)).get("save_file") +f'_accuracy{round(test_acc, 3)*1000}.keras')

def process_models():

    active_models_indexs = []
    active_models = []

    # Determining active models
    for i in range(1,6):
        if(models.get("model_" + str(i)).get("active")):
            active_models_indexs.append(i)

    # Making models for actives
    for index in active_models_indexs:
        model = make_model(index)
        model.summary()
        active_models.append(model)

    # Insert Wen's function here
    (data, labels) = getParticipantsExperiments(4, 1)

    for i in range(len(active_models)):
        modelNum = active_models_indexs[i]
        train_test_model(data, labels, active_models[i], modelNum)

def main():
    process_models()

main()