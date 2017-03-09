#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import preprocess


if __name__ == '__main__':
    data = np.load(preprocess.root_path + "/data")
    labels = np.load(preprocess.root_path + "/emotions")
    pca = preprocess.pca(data - data.mean(0))
    pcadata = pca.transform(data)
    # Выбрасываем неопределенные эмоции
    data = data[labels != -1]
    labels = labels[labels != -1]
    # Выбрасываем неприязнь, слишком мало данных
    data = data[labels != 2]
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
        restricted_data = data [labels == i]
        np.random.shuffle(restricted_data)
        spl = np.split(restricted_data[0:mincount],[training_count,training_count+test_count])
        training_data.append(spl[0])
        training_labels += [i] * training_count
        test_data.append(spl[1])
        test_labels += [i] * test_count
        validation_data.append(spl[2])
        validation_labels += [i] * validation_count
    training_data = np.array(training_data).reshape(-1, 136)
    test_data = np.array(test_data).reshape(-1, 136)
    validation_data = np.array(validation_data).reshape(-1, 136)
    training_labels = np.array(training_labels).flatten()
    test_labels = np.array(test_labels).flatten()
    validation_labels = np.array(validation_labels).flatten()


