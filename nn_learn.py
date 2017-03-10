#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import preprocess
import tflearn


if __name__ == '__main__':
    data = np.load(preprocess.root_path + "/data")
    labels = np.load(preprocess.root_path + "/emotions")
    pca = preprocess.pca(data - data.mean(0))
    pcadata = pca.transform(data)
    # Выбрасываем неопределенные эмоции
    pcadata = pcadata[labels != -1]
    labels = labels[labels != -1]
    # Выбрасываем неприязнь, слишком мало данных
    pcadata = pcadata[labels != 2]
    labels = labels[labels != 2]
    # Определяем метку с наименьшим количеством данных
    bincount = np.bincount(labels)
    minlabel = 0
    for i in range(labels.max()+1):
        if bincount[i] != 0 and bincount[minlabel] > bincount[i]:
            minlabel = i
    mincount = bincount [minlabel]
    # Разбиваем данные случайным образом на обучающее, тестовое и валидирующее множества в отношении 60:20:20
    training_count = int (mincount * 0.6)
    test_count = int (mincount * 0.2)
    validation_count = mincount - training_count - test_count
    # Выбираем нужное количество данных из каждой категории
    training_data = []
    training_labels = []
    test_data = []
    test_labels = []
    validation_data = []
    validation_labels = []
    for i in np.unique(labels):
        restricted_data = pcadata [labels == i]
        label = np.zeros(8)
        label[i] = 1
        np.random.shuffle(restricted_data)
        spl = np.split(restricted_data[0:mincount],[training_count,training_count+test_count])
        training_data.append(spl[0])
        training_labels += [label] * training_count
        test_data.append(spl[1])
        test_labels += [label] * test_count
        validation_data.append(spl[2])
        validation_labels += [label] * validation_count
    training_data = np.array(training_data).reshape(-1, 12)
    test_data = np.array(test_data).reshape(-1, 12)
    validation_data = np.array(validation_data).reshape(-1, 12)
    training_labels = np.array(training_labels).reshape(-1,8)
    test_labels = np.array(test_labels).reshape(-1,8)
    validation_labels = np.array(validation_labels).reshape(-1,8)


#    training_permut = np.random.permutation(training_data.shape[0])
#    test_permut = np.random.permutation(test_data.shape[0])
#    validation_permut = np.random.permutation(validation_data.shape[0])
#    training_data = training_data[training_permut]
#    test_data = test_data [test_permut]
#    validation_data = validation_data [validation_permut]
#    training_labels = training_labels[training_permut]
#    test_labels = test_labels [test_permut]
#    validation_labels = validation_labels [validation_permut]


# Собственно нейронная сеть

    input_layer = tflearn.input_data(shape=[None, 12])
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, 0.8)
    dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    softmax = tflearn.fully_connected(dropout2, 8, activation='softmax')
    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    top_k = tflearn.metrics.Top_k(3)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(training_data, training_labels, n_epoch=400, validation_set=(validation_data, validation_labels),
              show_metric=True,shuffle = True, run_id="dense_model")


