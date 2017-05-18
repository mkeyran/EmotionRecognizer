#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tflearn
import tensorflow as tf

import base_model

neural_net1_pca = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [64, 64],
    "use_dropout": True,
    "model_name": "nn_pca_64x64_lr_0.5",
    "learning_rate": 0.5,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net2_pca = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [128, 128],
    "use_dropout": True,
    "model_name": "nn_pca_128x128_lr_0.5",
    "learning_rate": 0.5,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}


neural_net3_pca = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [256, 256],
    "use_dropout": True,
    "model_name": "nn_pca_256x256_lr_0.5",
    "learning_rate": 0.5,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net4_pca = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [64, 64, 64],
    "use_dropout": True,
    "model_name": "nn_pca_64x64x64_lr_0.5",
    "learning_rate": 0.5,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net5_pca = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [128, 128, 128],
    "use_dropout": True,
    "model_name": "nn_pca_128x128x128_lr_0.5",
    "learning_rate": 0.5,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}


neural_net6_pca = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [256, 256, 256],
    "use_dropout": True,
    "model_name": "nn_pca_256x256x256_lr_0.5",
    "learning_rate": 0.5,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net1_pca_lr08 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [64, 64],
    "use_dropout": True,
    "model_name": "nn_pca_64x64_lr_0.8",
    "learning_rate": 0.8,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net2_pca_lr08 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [128, 128],
    "use_dropout": True,
    "model_name": "nn_pca_128x128_lr_0.8",
    "learning_rate": 0.8,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}


neural_net3_pca_lr08 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [256, 256],
    "use_dropout": True,
    "model_name": "nn_pca_256x256_lr_0.8",
    "learning_rate": 0.8,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net4_pca_lr08 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [64, 64, 64],
    "use_dropout": True,
    "model_name": "nn_pca_64x64x64_lr_0.8",
    "learning_rate": 0.8,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net5_pca_lr08 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [128, 128, 128],
    "use_dropout": True,
    "model_name": "nn_pca_128x128x128_lr_0.8",
    "learning_rate": 0.8,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}


neural_net6_pca_lr08 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [256, 256, 256],
    "use_dropout": True,
    "model_name": "nn_pca_256x256x256_lr_0.8",
    "learning_rate": 0.8,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net1_pca_lr03 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [64, 64],
    "use_dropout": True,
    "model_name": "nn_pca_64x64_lr_0.3",
    "learning_rate": 0.3,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net2_pca_lr03 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [128, 128],
    "use_dropout": True,
    "model_name": "nn_pca_128x128_lr_0.3",
    "learning_rate": 0.3,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}


neural_net3_pca_lr03 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [256, 256],
    "use_dropout": True,
    "model_name": "nn_pca_256x256_lr_0.3",
    "learning_rate": 0.3,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net3_5_pca_lr03 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [512, 512],
    "use_dropout": True,
    "model_name": "nn_pca_512x512_lr_0.3",
    "learning_rate": 0.3,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}


neural_net4_pca_lr03 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [64, 64, 64],
    "use_dropout": True,
    "model_name": "nn_pca_64x64x64_lr_0.3",
    "learning_rate": 0.3,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}

neural_net5_pca_lr03 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [128, 128, 128],
    "use_dropout": True,
    "model_name": "nn_pca_128x128x128_lr_0.3",
    "learning_rate": 0.3,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}


neural_net6_pca_lr03 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [256, 256, 256],
    "use_dropout": True,
    "model_name": "nn_pca_256x256x256_lr_0.3",
    "learning_rate": 0.3,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}


neural_net1_pca_lr01 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [64, 64],
    "use_dropout": True,
    "model_name": "nn_pca_64x64_lr_0.1",
    "learning_rate": 0.1,
    "learning_rate_decay":0.96,
    "metric_top_k": 1
}



neural_net1_nonpca = {
    "num_features": 136,
    "num_labels": 8,
    "num_layers": 2,
    "num_neurons": [64, 64],
    "use_dropout": True,
    "model_name": "model1_non_pca",
    "learning_rate": 0.1,
    "learning_rate_decay":0.96,
    "metric_top_k": 2
}


neural_net3_non_pca = {
    "num_features": 136,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [2048, 2048, 2048],
    "use_dropout": True,
    "model_name": "model_non_pca_3_layers_2048",
    "learning_rate": 1.0,
    "learning_rate_decay": 0.96,
    "metric_top_k": 1
}


neural_net2_non_pca = {
    "num_features": 136,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [1024*4, 4096*2, 1024*4],
    "use_dropout": True,
    "model_name": "model_non_pca_3_layers_4096_8192_4096_lr0.3",
    "learning_rate": 0.2,
    "learning_rate_decay": 0.96,
    "metric_top_k": 1,
    "batch_size": 512
}

neural_net2_pca1 = {
    "num_features": 27,
    "num_labels": 8,
    "num_layers": 3,
    "num_neurons": [64, 64, 64],
    "use_dropout": True,
    "model_name": "model_pca_3_layers_256_lr1.5",
    "learning_rate": 1.5,
    "learning_rate_decay": 0.96,
    "metric_top_k": 1
}

class NeuralNetwork(base_model.ClassificationModel):

    def __init__(self, params):
        self.params = params
        self.name = params["model_name"]
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
        # Regression using SGD with learning rate decay and Top-k accuracy
        sgd = tflearn.SGD(learning_rate=params["learning_rate"], lr_decay=params["learning_rate_decay"] , decay_step=1000)
        top_k = tflearn.metrics.Top_k(params["metric_top_k"])
        accuracy_metric = tflearn.metrics.Accuracy()
        net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                                 loss='categorical_crossentropy')

        # Training
        tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)
        with tf.device('/gpu:1'):
            self.model = tflearn.DNN(net, tensorboard_verbose=0,
                                     best_checkpoint_path="data/Models/"+self.params["model_name"],
                                     max_checkpoints=5,
                                     best_val_accuracy=0.6)

    def train(self, training_data, training_labels, validation_data, validation_labels, n_epoch):
        eye = np.eye(self.params["num_labels"])
        n_training_labels = eye[training_labels]
        n_validation_labels = eye[validation_labels]
        with tf.Graph().as_default():
            with tf.device('/gpu:1'):
                self.model.fit(training_data, n_training_labels, n_epoch=n_epoch, validation_set=(validation_data, n_validation_labels),
                           show_metric=True,
                           batch_size = self.params["batch_size"] if "batch_size" in self.params else None,
                           shuffle=True,
                           run_id=self.params["model_name"] + "sess")

    def save(self):
        self.model.save("data/Models/"+self.params["model_name"])

    def load(self):
        self.model.load("data/Models/"+self.params["model_name"])

    def predict(self, X):
        return self.model.predict(X)


def accuracy(model, test_data, test_labels):
    predicted = np.array(model.predict(test_data)).argmax(axis=1)
    actual = test_labels
    true, false = np.count_nonzero(
        predicted == actual), np.count_nonzero(predicted != actual)
    return true / (true + false)


if __name__=='__main__':
    import pickle
    dat = pickle.load(open("data/TrainingData/pickled_generated_sets",'rb'))
    pca_dat = pickle.load(open("data/TrainingData/pickled_generated_sets_pca",'rb'))