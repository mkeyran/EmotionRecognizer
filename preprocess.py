1  # !/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import dlib
import sklearn.decomposition
import pickle
import mappings
import copy
# Emotion tags in ascending order
emotions = ['neutral', 'anger', 'contempt', 'disgust',
            'fear', 'happiness', 'sadness', 'surprise']

root_path = "/home/keyran/Documents/Teaching/2016/Дипломники/Datasets/Ck_plus/"
images_path = root_path + "cohn-kanade-images"
emotion_labels_path = root_path + "Emotion"

cascadePath = "data/Cascades/haarcascade_frontalface_alt.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)

frontal_face_detector = dlib.get_frontal_face_detector()

pose_model = dlib.shape_predictor("data/ShapePredictor/shape_predictor_68_face_landmarks.dat")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

class Face:

    def __init__(self, filepath=None, image=None, rectangle=None, label=None):
        if (filepath is None and image is None):
            raise ValueError("You must scpecify either filepath or image")
        self._filepath = filepath
        self._image = image
        self._tilt = None
        self._milestones = None
        self._center = None
        self._emotions = None
        self.label = label
        self.rectangle = rectangle

    def image(self):
        if not self._image is None:
            return self._image
        else:
            return cv2.imread(self._filepath)

    def milestones(self):
        if self.rectangle is None:
            raise Exception("The face rectangle hasn't been set")
        if self._milestones is None:
            gray_image = cv2.cvtColor(self.image(), cv2.COLOR_RGB2GRAY)
            milestones = pose_model(gray_image, self.rectangle)
            arr = np.ndarray((milestones.num_parts, 2))
            for i in range(milestones.num_parts):
                arr[i, 0] = milestones.part(i).x
                arr[i, 1] = milestones.part(i).y
            self._milestones = arr
        return self._milestones

    @staticmethod
    def fabric(filepath=None, image=None, label=None):
        if (filepath is None and image is None):
            raise ValueError("You must scpecify either filepath or image")
        faces = []
        if filepath:
            image = cv2.imread(filepath)
        faces = frontal_face_detector(image)
        if len(faces) != 0:
            return [Face(filepath=filepath, image=image if not filepath else None,
                         rectangle=face, label=label) for face in faces]
        else:
            faces = faceCascade.detectMultiScale(image)
            if len(faces) != 0:
                ret = []
                for face in faces:
                    x = face[0]
                    y = face[1]
                    x1 = x + face[2]
                    y1 = y + face[3]
                    ret.append(dlib.rectangle(
                        int(x), int(y), int(x1), int(y1)))
                return [Face(filepath=filepath, image=image if not filepath else None,
                             rectangle=face, label=label) for face in ret]
            else:
                return []

    def tilt(self):
        if self._tilt is None:
            tilt = 180 - np.arctan2(self.milestones()[45][1] - self.milestones()[39][1],
                                    self.milestones()[39][0] - self.milestones()[45][0]) * 180 / np.pi
            self._tilt = tilt if tilt < 180 else tilt - 360
        return self._tilt

    def center(self):
        if self._center is None:
            self._center = (self.milestones()[39][0] + self.milestones()[45][0] / 2,
                            self.milestones()[39][1] + self.milestones()[45][1] / 2)
        return self._center


class FaceSet:


    def __init__(self, faces, maps = []):
        self.faces = faces
        self.permutation = None
        self.mappings = maps

    def generate_training_data(self):
        data = np.array([f.milestones() for f in self.faces])
        labels = np.array([f.label for f in self.faces])  
        for mapping in self.mappings:
            data, labels = mapping.training_mapping(data, labels)
        return data.reshape(len(labels),-1), labels
    
    def generate_data(self):
        data = np.array([f.milestones() for f in self.faces])
        data_len = len(data)
        for mapping in self.mappings:
            data = mapping.classification_mapping(data)
        return data.reshape(data_len,-1)

    def generate_sets(self, train_size=0.6, validation_size=0.2, test_size=0.2,
                      permutation=None):
        points,labels = self.generate_training_data()
        if permutation:
            self.permutation = permutation
        if self.permutation is None:
            self.permutation = np.random.permutation(points.shape[0])
        training_count = int(points.shape[0] * train_size)
        validation_count = int(points.shape[0] * validation_size)
        training_data, validation_data, test_data = np.split(points[self.permutation],
                                                             [training_count, validation_count + training_count])
        training_labels, validation_labels, test_labels = np.split(labels[self.permutation],
                                                                   [training_count, validation_count + training_count])
        return {"train_data": training_data,
                "train_labels": training_labels,
                "valid_data": validation_data,
                "valid_labels": validation_labels,
                "test_data": test_data,
                "test_labels": test_labels}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


def get_emotion(subject, series):
    path = emotion_labels_path + "/" + subject + "/" + series
    if not os.path.exists(path):
        return -1
    files = os.listdir(path)
    if len(files) == 0:
        return -1
    with open(path + "/" + files[0], 'r') as fin:
        str_ = fin.readline()
        return int(float(str_))


def load_all():
    faces = []
    for p1 in os.scandir(images_path):
        subject = p1.name
        if not p1.is_dir():
            continue
        for p2 in os.scandir(p1.path):
            if not p2.is_dir():
                continue
            num = p2.name
            for p3 in os.scandir(p2.path):
                if not p3.is_file():
                    continue
                image = p3.path
              #  print (image)
                # Начальные изображения - нейтральные
                if int(image.rsplit("_")[-1][:-4]) > 8:
                    emotion = get_emotion(subject, num)
                else:
                    emotion = 0
                faces.append(Face.fabric(filepath=p3.path, label=emotion)[0])
                print(len(faces))
    return FaceSet(faces)

if __name__ == '__main__':
    try:
        face_set = FaceSet.load("data/TrainingData/training_data.dat")
    except:
        face_set = load_all()
    pca_face_set = copy.deepcopy(face_set)
    face_set.mappings = (mappings.DropContemptMapping(), 
                         mappings.NormalizeMapping(), 
                         mappings.ImageMirrorMapping()
                         )

    pca = mappings.PCAMapping()

    pca_face_set.mappings = (mappings.DropContemptMapping(), 
                         mappings.NormalizeMapping(), 
                         mappings.ImageMirrorMapping(),
                         pca)
    face_set.save("data/TrainingData/training_data.dat")
    pickle.dump(pca,open("data/TrainingData/pcamapping.dat",'wb'))
    if(input("Generate training data? (yes/no) ")=="yes"):
        dat = face_set.generate_sets()
        pickle.dump(dat, open("data/TrainingData/pickled_generated_sets",'wb'))
        pca_dat = pca_face_set.generate_sets()
        pickle.dump(pca_dat, open("data/TrainingData/pickled_generated_sets_pca",'wb'))
        pickle.dump(pca, open("data/TrainingData/pcamapping.dat", 'wb'))
