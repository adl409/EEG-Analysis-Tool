import tensorflow as tf
import numpy as np
from dataParser import *
import time
from sklearn.model_selection import train_test_split

def main():
    makeIndividualModel(1, 2)

def makeIndividualModel(interval, individual):


    print("Gathering Data... \n")
    startTime = time.time()

    # Training on all participants
    #(data, labels) = getAllData(interval)

    # Training on single participant
    (data, labels) = getParticipantsExperiments(individual, interval)

    data_train, data_test, labels_train, labels_test = train_test_split(data,labels , random_state=104,test_size=0.20, shuffle=True)

    dataGatheredTime = time.time()

    print("All data has been gathered... \n Time elapsed: ", dataGatheredTime-startTime, '\n')

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(128, 65)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    print("Model Compiled... \nStarting Training...\n")

    model.fit(data_train, labels_train, epochs=20)

    modelFitTime = time.time()

    print("Model Trained... \nTime elapsed: ", modelFitTime - dataGatheredTime, '\n')

    test_loss, test_acc = model.evaluate(data_test,  labels_test, verbose=2)

    print('\nTest accuracy:', test_acc, '\n')

    model.save(f'./results/result_interval{interval}_person{individual}_accuracy{round(test_acc, 3)*1000}.h5')

main()