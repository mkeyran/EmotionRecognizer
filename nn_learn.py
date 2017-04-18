#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tflearn
import tensorflow as tf

neural_net1_pca = {
    "num_features": 12,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [64, 64],
    "use_dropout": True,
    "model_name": "model1"
}

neural_net1_nonpca = {
    "num_features": 136,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [64, 64],
    "use_dropout": True,
    "model_name": "model1_non_pca"
}


class NeuralNetwork(object):

    def __init__(self, params):
        self.params = params
        input_layer = tflearn.input_data(shape=[None, params["num_features"]])
        prev_layer = input_layer
        for i in range(params["num_layers"]):
            dense = tflearn.fully_connected(prev_layer, params["num_neurons"][i], activation='relu',
                                            regularizer='L2', weight_decay=0.001)
            prev_layer = dense
            if params["use_dropout"]:
                dropout1 = tflearn.dropout(prev_layer, 0.8)
                prev_layer = dropout1
        softmax = tflearn.fully_connected(
            prev_layer, params["num_labels"], activation='softmax')
        # Regression using SGD with learning rate decay and Top-3 accuracy
        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        top_k = tflearn.metrics.Top_k(3)
        net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                                 loss='categorical_crossentropy')

        # Training
        self.model = tflearn.DNN(net, tensorboard_verbose=0)

    def train(self, training_data, training_labels, validation_data, validation_labels, n_epoch):
        eye = np.eye(self.params["num_labels"])
        n_training_labels = eye[training_labels]
        n_validation_labels = eye[validation_labels]
        with tf.Graph().as_default():
            self.model.fit(training_data, n_training_labels, n_epoch=n_epoch, validation_set=(validation_data, n_validation_labels),
                           show_metric=True, shuffle=True, run_id=self.params["model_name"] + "sess")

    def save(self):
        self.model.save("data/Models/"+self.params["model_name"])

    def load(self):
        self.model.load("data/Models/"+self.params["model_name"])

    def predict(self, X):
        return self.model.predict(X)


def accuracy(model, test_data, test_labels):
    predicted = np.array(model.predict(test_data)).argmax(axis=1)
    actual = test_labels.argmax(axis=1)
    true, false = np.count_nonzero(
        predicted == actual), np.count_nonzero(predicted != actual)
    return true / (true + false)
