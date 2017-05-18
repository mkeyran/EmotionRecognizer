import numpy as np
import pickle
import sklearn.metrics

class ClassificationModel:

    def __init__(self, name, **kwargs):
        self.name = name
        self.args = kwargs

    def train(self, training_data: np.ndarray, training_labels: np.ndarray):
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        pass

    def save(self):
        pickle.dump(self, open("data/Models/" + self.name, "wb"))

    @staticmethod
    def load(name: str):
        return pickle.load(open("data/Models/" + name, "rb"))


def accuracy(model: ClassificationModel, test_data: np.ndarray, test_labels: np.ndarray) -> float:
    """
    Вычисляет точность модели 
    Parameters
    ----------
    model Модель
    test_data Проверочные данные
    test_labels Тестовые данные

    Returns
    -------
    Точность модели, правильно классифицированные примеры / общее количество примеров

    """
    predicted = np.array(model.predict(test_data)).argmax(axis=1)
    return sklearn.metrics.accuracy_score(test_labels, predicted)


def confusion_matrix(model: ClassificationModel, test_data: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
    predicted = np.array(model.predict(test_data)).argmax(axis=1)
    return sklearn.metrics.confusion_matrix(test_labels, predicted)


def precision(model: ClassificationModel, test_data: np.ndarray, test_labels: np.ndarray) -> float:
    predicted = np.array(model.predict(test_data)).argmax(axis=1)
    return sklearn.metrics.precision_score(test_labels, predicted, average='macro')


def recall(model: ClassificationModel, test_data: np.ndarray, test_labels: np.ndarray) -> float:
    predicted = np.array(model.predict(test_data)).argmax(axis=1)
    return sklearn.metrics.recall_score(test_labels, predicted, average='macro')


def f1_score(model: ClassificationModel, test_data: np.ndarray, test_labels: np.ndarray) -> float:
    predicted = np.array(model.predict(test_data)).argmax(axis=1)
    return sklearn.metrics.f1_score(test_labels, predicted, average='macro')
