from layerClasses import *
import tensorflow as tf
import numpy as np
from dataParser import *
from sklearn.model_selection import train_test_split
import os
from contextlib import redirect_stdout


defaults = {
    "standard": [Flatten(0), Dense(0, 64, "relu"), Dense(0,64,"relu")],
    "cnn": [Convolution_2d(0,filters=16, kernel_size=3), Max_Pool_2d(0,pool_size=(2,2)),Convolution_2d(0,filters=32, kernel_size=3), Max_Pool_2d(0,pool_size=(2,2)),Convolution_2d(0,filters=64, kernel_size=3), Max_Pool_2d(0,pool_size=(2,2)), Flatten(0), Dense(0,64), Dense(0,32) ],
    "rnn": [SimpleRNN(0,16), Dense(0, 64)]
}

models = {
    "input_parameters": {
        "input_shape1" : 128,
        "input_shape2" : 65,
        "root_directory" : "./Van250Tot0",
        "normalized" : False
    },
    "test_parameters": {
        "test_split" : 0.20,
        "epochs" : 10,
        "shuffle" : True
    },
    "model_1" : {
        "save_model" : True, 
        "save_file" : "saveFile2.h",
        "active": True,
        "layers": #[SimpleRNN(0, 2)]
        defaults["rnn"]
        #[Max_Pool_2d(2, pool_size=(2,2)),SimpleRNN(0,2), Dense(0,10), Flatten(0)]
    }
}

#presetCNN = []
def parseData(rootPath):

    # File Pattern
    #-Root
    # --P1
    #   --L1
    #      --data.csv
    #   --L2
    #   --L3
    # --P2
    #   --L1
    #   --L2
    #   --L3

    data = []
    labels = []

    visited = []
    label_nums = {}
    counter = 0

    # Getting all particpants
    for participant in os.listdir(rootPath):

        fullPartPath = os.path.join(rootPath, participant)
        
        # Reading all labels for given participant
        for label in os.listdir(fullPartPath):
            
            # Adding label to visited label set if not yet added
            if label not in visited:
                visited.append(label)
                # need labels to be integer values
                # storing int values in dict
                label_nums[label] = counter
                counter += 1

            fullLabelPath = os.path.join(fullPartPath, label)

            for file in os.listdir(fullLabelPath):
                fullFilePath = os.path.join(fullLabelPath, file)
                try:
                    data.append(np.genfromtxt(fullFilePath, delimiter=","))
                    labels.append(label_nums.get(label))
                except:
                    print("File: '", fullFilePath, "' not able to be read")


    # Combining data into single 3d array in form (Files, time, electrode)
    data = np.dstack(data)
    data = np.transpose(data, (2,0,1))
    # cast labels as np array
    labels = np.array(labels)

    print("Shape of Data: ", data.shape)
    print("Label Numbers: ", label_nums)

    return [(data, labels), label_nums]
    
            ##iterating through all files of given label
            #for file in 

            # Need output in form of 3d np array and list of labels

def make_model(models, numLabels):

    # Pulling layers from dict
    layers = models.get("model_1").get("layers") 

    # Creating model
    
    model = tf.keras.Sequential()

    # Configuring input layer
    inputShape1 = models.get("input_parameters").get("input_shape1")
    inputShape2 = models.get("input_parameters").get("input_shape2")
    # Fourth dim added to make compatible with 2d functions

    #Setting input size
    # If model has RNN layers, dont add batch size, else add batch size    
    rnn = False
    cnn = False
    for x in layers:
        if(x.layerType == "simplernn" or x.layerType == "lstm" or x.layerType == "gru"):
            rnn = True
        if(x.layerType == "convolution_2d" or x.layerType == "convolution_2d_transpose" or x.layerType == "max_pool_2d" or x.layerType == "average_pooling_2d"):
            cnn = True
    
    if(cnn and rnn):
        print("ERROR: Invalid Layer Configuration")
        return tf.keras.Sequential()
    else:
        if rnn: 
            model.add(tf.keras.layers.Input(shape=(int(inputShape1), int(inputShape2))))

        else:
            model.add(tf.keras.layers.Input(shape=(int(inputShape1), int(inputShape2), 1)))


    addFlat = True    

    # Adding layers based on dict
    for layer in layers:
        
        match(layer.layerType):
            case "dense":
                model.add(tf.keras.layers.Dense(layer.units, activation = layer.activation, use_bias = layer.use_bias, kernel_initializer=layer.kernel_initializer, bias_initializer = layer.bias_initializer))
            
            case "flatten": 
                model.add(tf.keras.layers.Flatten())
                addFlat = False

            case "zero_padding_2d":
                model.add(tf.keras.layers.ZeroPadding2D(padding = layer.padding))

            case "average_pooling_2d":
                model.add(tf.keras.layers.AveragePooling2D(pool_size = layer.pool_size, strides = layer.strides, padding = layer.padding))

            case "max_pool_2d":
                model.add(tf.keras.layers.MaxPooling2D(pool_size=layer.pool_size, strides = layer.strides, padding = layer.padding))

            case "convolution_2d":
                print(layer.filter)
                model.add(tf.keras.layers.Conv2D(filters = layer.filter, kernel_size = layer.kernel_size, strides = layer.strides, padding = layer.padding, dilation_rate = layer.dialation_rate, groups = layer.groups, activation = layer.activation, use_bias = layer.use_bias, kernel_initializer = layer.kernel_initializer, bias_initializer = layer.bias_initializer))

            case "convolution_2d_transpose":
                model.add(tf.keras.layers.Conv2DTranspose(filters = layer.filter, kernel_size = layer.kernel_size, strides = layer.strides, padding = layer.padding, dilation_rate = layer.dialation_rate, groups = layer.groups, activaiton = layer.activation, use_bias = layer.use_bias, kernel_initializer = layer.kernel_initializer, bias_initializer = layer.bias_initializer))
            
            case "simplernn":
                model.add(tf.keras.layers.SimpleRNN(units = layer.units, activation = layer.activation, use_bias = layer.use_bias, kernel_initializer = layer.kernel_initializer, recurrent_initializer = layer.recurrent_initializer, bias_initializer = layer.bias_initializer, dropout = layer.dropout, recurrent_dropout = layer.recurrent_dropout))

            case "lstm":
                model.add(tf.keras.layers.LSTM(units = layer.units, activation = layer.activation, use_bias = layer.use_bias, kernel_initializer = layer.kernel_initializer, recurrent_initializer = layer.recurrent_initializer, bias_initializer = layer.bias_initializer, dropout = layer.dropout, recurrent_dropout = layer.recurrent_dropout, seed = layer.seed, unit_forget_bias = layer.unit_forget_bias, recurrent_activation = layer.recurrent_activation))
                
            case "gru":
                 model.add(tf.keras.layers.GRU(units = layer.units, activation = layer.activation, use_bias = layer.use_bias, kernel_initializer = layer.kernel_initializer, recurrent_initializer = layer.recurrent_initializer, bias_initializer = layer.bias_initializer, dropout = layer.dropout, recurrent_dropout = layer.recurrent_dropout, seed = layer.seed, unit_forget_bias = layer.unit_forget_bias, recurrent_activation = layer.recurrent_activation, reset_after = layer.reset_after))

            case _ :
                print("Error Determining layer type...")
    
    if addFlat:
        model.add(tf.keras.layers.Flatten())
    # Adding output layer with number of outputs corresponding to number of labels
    model.add(tf.keras.layers.Dense(numLabels))

    return model

def train_test_model(data, labels, model, models):

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels , random_state=104,test_size=models.get("test_parameters").get("test_split"), shuffle=models.get("test_parameters").get("shuffle"))
     
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
     
    model.fit(data_train, labels_train, epochs=models.get("test_parameters").get("epochs"))

    test_loss, test_acc = model.evaluate(data_test,  labels_test, verbose=2)

    if(models.get("model_1").get("save_model")):
        model.save(f'./'+ models.get("model_1").get("save_file") +f'_accuracy{round(test_acc, 3)*100}.keras')

    return test_loss, test_acc

def process_model(models):

    # Collecting input data
    parseInfo = parseData(models.get("input_parameters").get("root_directory"))
    (data, labels) = parseInfo[0]
    labelDict = parseInfo[1]

    # Making model
    model = make_model(models, len(labelDict))
    model.summary()  

    # Training model
    loss, accuracy = train_test_model(data, labels, model, models)

    # Creating result summary
    with open("result.txt", "w+") as result:
        
        result.write("---Labels---\n\n")
        result.write("(Label: value)\n")
        for key in labelDict:
            result.write(str(key))
            result.write(": ")
            result.write(str(labelDict[key]))
            result.write("\n")

        result.write("\n---Model Summary---\n\n")
        
        with redirect_stdout(result):
            model.summary()

        result.write("\n\n---Testing Summary---\n\n")
        result.write("Testing Accuracy: ")
        result.write(str(round(accuracy, 4)*100)[0:5])
        result.write("%")
        result.write("\nTesting Loss: ")
        result.write(str(loss))

        if(models.get("model_1").get("save_model")):
            result.write("\n\nModel saved to: ")
            result.write(models.get("model_1").get("save_file"))
            result.write("\n")

def main():
    process_model(models)

main()