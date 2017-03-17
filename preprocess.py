1#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import dlib
import sklearn.decomposition
import pickle


# Emotion tags in ascending order
emotions = ['neutral', 'anger', 'contempt', 'disgust',
            'fear', 'happiness', 'sadness', 'surprise']

root_path = "/home/keyran/Documents/Teaching/2016/Дипломники/Datasets/Ck_plus/"
images_path = root_path + "cohn-kanade-images"
emotion_labels_path = root_path + "Emotion"

cascadePath = "haarcascade_frontalface_alt.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)

frontal_face_detector = dlib.get_frontal_face_detector()

pose_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Требует 0 матожидания => данные перед применением необходимо центрировать (arr = arr - arr.mean(0))
def pca (arr, keep_variance = 0.95):
    pca = sklearn.decomposition.PCA(n_components=keep_variance, svd_solver='full')
    pca.fit(arr)
    return pca



class Face:
    def __init__(self, filepath = None, image = None, rectangle = None, label = None):
        if (filepath is None and image is None):
            raise ValueError ("You must scpecify either filepath or image")
        self._filepath = filepath 
        self._image = image
        self._tilt = None
        self._milestones = None
        self._center = None
        self._emotions = None
        self.label = label
        self.rectangle = rectangle
        
    def image(self):
        if self._image:
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
    def fabric (filepath = None, image = None, label = None):
        if (filepath is None and image is None):
            raise ValueError ("You must scpecify either filepath or image")
        faces = []
        if filepath: image = cv2.imread(filepath)
        faces = frontal_face_detector(image)
        if len(faces) != 0:
            return [Face(filepath = filepath, image = image if not filepath else None, 
                         rectangle = face, label = label) for face in faces]
        else:
            faces = faceCascade.detectMultiScale(image)
            if len(faces) != 0:
                ret = []
                for face in faces:
                    x = face[0]
                    y = face[1]
                    x1 = x + face[2]
                    y1 = y + face[3]
                    ret.append(dlib.rectangle(int(x),int(y),int(x1),int(y1)))
                return [Face(filepath = filepath, image = image if not filepath else None, 
                         rectangle = face, label = label) for face in ret]
            else:
                return []
        
        
    def tilt(self):
        if self._tilt is None: 
            tilt = 180 - np.arctan2( self.milestones()[45][1]- self.milestones()[39][1], 
                                     self.milestones()[39][0]- self.milestones()[45][0])*180/np.pi
            self._tilt = tilt if tilt<180 else tilt - 360
        return self._tilt
    
    def center(self):
        if self._center is None:
            self._center = (self.milestones()[39][0] + self.milestones()[45][0] / 2, 
                            self.milestones()[39][1] + self.milestones()[45][1] / 2)
        return self._center

class FaceSet:
    # Отображение номеров точек при зеркалировании
    mirrormap = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,
             26,25,24,23,22,21,20,19,18,17,
             27,28,29,30,
             35,34,33,32,31,
             45,44,43,42,47,46,
             39,38,37,36,41,40,
             54,53,52,51,50,49,48,
             59,58,57,56,55,
             64,
             63,62,61,
             60,
             67,66,65]
        
    def __init__(self, faces):
        self.faces = faces
    
    def normalisation(self,arr):
        max = np.max(arr, 0)
        min = np.min(arr, 0)
        return (arr-min)/(max - min)-0.5

    def standardization(self,arr):
    # Переводим массив в новый массив с 0 матожиданием и единичной дисперсией
        mean = arr.mean(0)
        variance = arr.var(0)
        return (arr-mean)/np.sqrt(variance), mean, variance

    def undo_standartization(self,arr, mean, var):
        return arr * np.power(var, 2) + mean
    
    def mirror(self, arr):
        return np.apply_along_axis(lambda x: [-x[0], x[1]], 1, arr)[self.mirrormap]
    
    def generate_data(self):
        points = np.array([normalisation(face.milestones()).flatten() for face in self.faces]+
                          [normalisation(mirror(face.milestones())).flatten() for face in self.faces]
                           )
        labels = np.array([face.label for face in self.faces] * 2)
        return points, labels
    
    def generate_sets(self,train_size = 0.6, validation_size = 0.2, test_size = 0.2,
                      permutation = None, pca_enabled = False):
        points, labels = generate_data()
        if pca_enabled:
            self.pca = pca(points)
            points = pca.transform(points)
        if permutation:
            self.permutation = permutation
        if not self.permutation:
            self.permutation = np.random.permutation(points.shape[0])
        training_count = points.shape[0]*train_size
        validation_count = points.shape[0]*validation_size
        test_count = point.shape[0]-test_count-validation_count
        training_data, validation_data, test_data=np.split(points, 
                [training_count, validation_count, test_count])
        training_labels, validation_labels, test_labels=np.split(labels, 
                [training_count, validation_count, test_count])
        return  ( [training_data, training_labels],
                  [validation_data, validation_labels],
                  [test_data, test_labels])
                
        
def get_emotion (subject: str, series: str):
    path = emotion_labels_path + "/" + subject + "/" + series
    if not os.path.exists(path):
        return -1
    files = os.listdir(path)
    if len(files) == 0:
        return -1
    file = files[0]
    with open (path + "/" + files[0],'r') as fin:
        str = fin.readline()
        return int(float(str))

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
                if int(image.rsplit("_")[-1][:-4]) > 8: # Начальные изображения - нейтральные
                    emotion = get_emotion(subject, num)
                else:
                    emotion = 0
                faces.append(Face.fabric(filepath = p3.path, label = emotion)[0])
                print (len(faces))
    return FaceSet(faces)




def save_data(labels, data, name):
    np.save(name + "_data", data)
    np.save(name + "_labels", labels)
    
def load_saved_data(name):
    data = np.load(name + "_data")
    labels = np.load(name + "_labels")
    return labels, data
    