
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from random import randint

import numpy as np

import pickle

import base_model


class DecisionForest(base_model.ClassificationModel):
    """ Класс, предоставляющий модель решающего леса (ансамбля решающих деревьев)
        Ансамбль генерирует заданное количество деревьев случайной глубины.
    """

    def __init__(self, name="forest", trees=80, min_tree_depth=5, max_tree_depth=25):
        """

        Parameters
        ----------
        name Метка модели
        trees 
            Количество деревьев в лесу
        min_tree_depth 
            Минимальная глубина дерева
        max_tree_depth 
            Максимальная глубина дерева
        """
        self.name = name
        self.trees = trees
        self.min_tree_depth = min_tree_depth
        self.max_tree_depth = max_tree_depth
        self.ensemble = []

    def train(self, training_data, training_labels):
        """
        Обучение леса
        Parameters
        ----------
        training_data 
            Данные для обучения
        training_labels
            Метки
        """
        self.ensemble = [tree.DecisionTreeClassifier(max_depth=randint(self.min_tree_depth, self.max_tree_depth))
                         for _ in range(self.trees)]
        for t in self.ensemble:
            t.fit(training_data, training_labels)

    def save(self):
        pickle.dump(self, open("data/Models/" + self.name, "wb"))

    def load(self):
        pickle.loads(self, open("data/Models/" + self.name, "rb"))

    def predict(self, data):
        """
        Классификация данных. 
        Parameters
        ----------
        data 
            Данные для классификации
            
        Returns
        -------
            Двумерный массив np.ndarray, в котором по строкам записаны элементы данных, по столбцам --- вероятность 
            принадлежности этого элемента к n-тому классу
        """
        prediction = list(map(lambda x: x.predict(data), self.ensemble))

        def max_p(d):
            """
                Функция классификации для одного элемента данных
            """
            #  Группируем данные и получаем количество вхождений в каждую из групп
            pred, count = np.unique(d, return_counts=True)
            z = np.zeros(d.shape)
            # Вероятность класса = количество предсказаний этого класса / количество деревьев в лесу
            z[pred] = count / self.trees
            return z
        # Находим вероятности классов для каждого из элементов данных
        return np.apply_along_axis(lambda x: max_p(x), 1, np.dstack(prediction)[0])


class DecisionForestSkLearn:
    """ Класс, предоставляющий модель решающего леса (ансамбля решающих деревьев)
        Ансамбль генерирует заданное количество деревьев случайной глубины.
    """

    def __init__(self, name="forest", trees=80, min_tree_depth=5, max_tree_depth=25):
        """

        Parameters
        ----------
        name Метка модели
        trees 
            Количество деревьев в лесу
        min_tree_depth 
            Минимальная глубина дерева
        max_tree_depth 
            Максимальная глубина дерева
        """
        self.name = name
        self.trees = trees
        self.min_tree_depth = min_tree_depth
        self.max_tree_depth = max_tree_depth
        self.forest = RandomForestClassifier(n_estimators=trees, max_depth=max_tree_depth)

    def train(self, training_data, training_labels):
        """
        Обучение леса
        Parameters
        ----------
        training_data 
            Данные для обучения
        training_labels
            Метки
        """
        self.forest.fit(training_data, training_labels)

    def save(self):
        pickle.dump(self, open("data/Models/" + self.name, "wb"))

    @staticmethod
    def load(name):
        return pickle.load(open("data/Models/" + name, "rb"))

    def predict(self, data):
        """
        Классификация данных. 
        Parameters
        ----------
        data 
            Данные для классификации

        Returns
        -------
            Двумерный массив np.ndarray, в котором по строкам записаны элементы данных, по столбцам --- вероятность 
            принадлежности этого элемента к n-тому классу
        """
        predicted_probabilities = self.forest.predict_proba(data)
        rearranged_probabilities = np.zeros((predicted_probabilities.shape[0], np.max(self.forest.classes_) + 1))
        rearranged_probabilities[:, self.forest.classes_] = predicted_probabilities
        return rearranged_probabilities

if __name__ == '__main__':
    dat = pickle.load(open("data/TrainingData/pickled_generated_sets", 'rb'))
    pca_dat = pickle.load(open("data/TrainingData/pickled_generated_sets_pca", 'rb'))
    #dec_forest_pca = DecisionForest("forest_pca")