#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import dlib
import sklearn.decomposition

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

def get_emotion (subject, series):
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

def find_faces(image):
    faces = frontal_face_detector(image)
    if len(faces) != 0:
        return faces
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
            return ret
        else:
            return None


def get_milestones(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = find_faces(gray_image)
    if faces is None:
        return []
    ret = []
    for face in faces:
        milestones = pose_model(gray_image, face)
        arr = np.ndarray((milestones.num_parts, 2))
        for i in range(milestones.num_parts):
            arr[i, 0] = milestones.part(i).x
            arr[i, 1] = milestones.part(i).y
        ret.append([face, arr])
    return ret

def normalisation(arr):
    max = np.max(arr, 0)
    min = np.min(arr, 0)
    return (arr-min)/(max - min)-0.5

def standardization(arr):
    # Переводим массив в новый массив с 0 матожиданием и единичной дисперсией
    mean = arr.mean(0)
    variance = arr.var(0)
    return (arr-mean)/np.sqrt(variance), mean, variance

def undo_standartization(arr, mean, var):
    return arr * np.power(var, 2) + mean

def mirror(arr):
    return np.apply_along_axis(lambda x: [-x[0], x[1]], 1, arr)[mirrormap]


def load_all():
    labels = []
    data = []
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
                if int(image.rsplit("_")[-1][:-4]) > 5: # Начальные изображения - нейтральные
                    emotion = get_emotion(subject, num)
                else:
                    emotion = 0
                limage = cv2.imread(path)
                if limage is None:
                    continue
                d = get_milestones(limage)
                if d is None: continue
                dn = normalisation(d[0][1])
                labels.append([emotion]*2)
                data.append(mirror(dn).flatten()) # Отражаем лицо относительно оси x=0.5, поскольку лица, вообще, симметричны
                data.append(dn.flatten())
                print (len(data))
             #   print (emotion)
    return np.array(labels).flatten(), np.array(data)


# Требует 0 матожидания => данные перед применением необходимо центрировать (arr = arr - arr.mean(0))
def pca (arr, keep_variance = 0.95):
    pca = sklearn.decomposition.PCA(n_components=keep_variance, svd_solver='full')
    pca.fit(arr)
    return pca

