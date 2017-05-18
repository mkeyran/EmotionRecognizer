#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:50:58 2017

@author: keyran
"""

import numpy as np
import sklearn


class Mapping(object):
    """ Базовый класс для операций преобразования данных. """

    def training_mapping(self, data, labels):
        """ Функция преобразования данных для обучения нейронной сети.

            Parameters
            ----------
            data: numpy.ndarray
                данные для преобразования.
            labels: numpy.ndarray
                метки для преобразований.

            Returns
            -------
            Пара из преобразованных данных и преобразованных меток.
        """
        pass

    def classification_mapping(self, data):
        """ Функция, преобразующая данные для классификации.

            Parameters
            ----------
            data : numpy.ndarray
                данные для преобразования

            Returns
            -------
            Преобразованные данные
        """
        pass


class ZoomAndTranslateMapping(Mapping):
    """ Класс, выполняющий перенос и масштабирование данных.

        Алгоритм: :math:`\fraq{X-d_min}{d_{max}-d_{min}}`
    """

    def training_mapping(self, data, labels):
        return self.classification_mapping(data), labels

    def classification_mapping(self, data):
        return  np.array([self.one_max(x) for x in data])
    
    def one_max (self,d):
        nd = d.reshape(-1,2)
        max_ = np.max(nd, 0)
        min_ = np.min(nd, 0)
        return ((nd - min_) / (max_ - min_) - 0.5)


class NormalizeMapping(Mapping):
    """ Класс, выполняющий нормализацию данных.

        Алгоритм: :math:`X-X.mean(axis=0)`
    """
    def __init__(self):
        self.mean = None

    def training_mapping(self, data, labels):
        self.mean = data.mean(axis = 0)
        return self.classification_mapping(data), labels

    def classification_mapping(self, data):
        return data-self.mean

class PCAMapping(Mapping):
    """ Класс, выполняющий анализ главных компонент для данных.
        Данные должны быть нормализованы перед применением преобразований,
        предоставляемых этим классом.

        Parameters
        ----------
        data: numpy.ndarray
            Данные для тренировки PCA
        keep_variance : Int, optional
            Количество оставляемой дисперсии.

        См. https://ru.wikipedia.org/wiki/Метод_главных_компонент
    """

    def __init__(self, keep_variance=0.95):
        self.keep_variance = keep_variance
        self.pca = None
        self.mean = None
        
    def pca_init (self, data):
        """
        Функция, инициализирующая PCA на обучающих данных.
        """
        #TODO: Обучать PCA только на обучающих данных
        self.mean = data.mean()
        data = data - self.mean
        self.pca = sklearn.decomposition.PCA(
            n_components=self.keep_variance, svd_solver='full')
        flatten_data = [d.flatten() for d in data]
        self.pca.fit(flatten_data)
    
    def training_mapping(self, data, labels):

        if self.pca is None:
            self.pca_init(data)
        return self.pca.transform(data.reshape(data.shape[0],-1)), labels

    def classification_mapping(self, data):
        data = data - self.mean
        return self.pca.transform(data.reshape(data.shape[0],-1))


class DropContemptMapping(Mapping):
    """ Класс, выбрасывающий из набора данных неприязнь (слишком мало обучающих примеров)"""

    def training_mapping(self, data, labels):
        new_data = data[labels != 2]
        new_labels = labels[labels != 2]
        return new_data, new_labels

    def classification_mapping(self, data):
        return data


class ImageMirrorMapping(Mapping):

    """ Класс, дополняющий набор данных отзеркалированными по горизонтали данными.
        Данные должны быть нормализованы так, чтобы центральная линия лица проходила через 0.
    """

    # Номера точек для отображения
    mirrormap = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                 27, 28, 29, 30,
                 35, 34, 33, 32, 31,
                 45, 44, 43, 42, 47, 46,
                 39, 38, 37, 36, 41, 40,
                 54, 53, 52, 51, 50, 49, 48,
                 59, 58, 57, 56, 55,
                 64,
                 63, 62, 61,
                 60,
                 67, 66, 65]

    def training_mapping(self, data, labels):
        new_data = np.concatenate([data, self.__mirror(data)])
        new_labels = np.concatenate([labels, labels])
        assert (new_data.shape [1:] == data.shape[1:])
        return new_data, new_labels

    def classification_mapping(self, data):
        return data

    def __mirror(self, data):
        return np.array([ np.apply_along_axis(lambda x: [-x[0], x[1]], 1, arr)[self.mirrormap] for arr in data])
