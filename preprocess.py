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

def find_face(image):
    faces = frontal_face_detector(image)
    if len(faces) != 0:
        return faces[0]
    else:
        faces = faceCascade.detectMultiScale(image)
        if len(faces) != 0:
            x = faces[0][0]
            y = faces[0][1]
            x1 = x + faces[0][2]
            y1 = y + faces[0][3]
            return dlib.rectangle(x,y,x1,y1)
        else:
            return None


def get_milestones(path):
    image = cv2.imread(path)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    face = find_face(gray_image)
    if face is None:
        return None
    milestones = pose_model(gray_image, face)
    arr = np.ndarray((milestones.num_parts, 2))
    for i in range(milestones.num_parts):
        arr[i, 0] = milestones.part(i).x
        arr[i, 1] = milestones.part(i).y
    return arr

def normalisation(arr):
    max = np.max(arr, 0)
    min = np.min(arr, 0)
    return (arr-min)/(max - min)

def standardization(arr):
    # Переводим массив в новый массив с 0 матожиданием и единичной дисперсией
    mean = arr.mean(0)
    variance = arr.var(0)
    return (arr-mean)/np.sqrt(variance), mean, variance

def undo_standartization(arr, mean, var):
    return arr * np.power(var, 2) + mean


def whitening (arr):
    pass


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
                print (image)
                if int(image.rsplit("_")[-1][:-4]) > 5: # Начальные изображения - нейтральные
                    emotion = get_emotion(subject, num)
                else:
                    emotion = 0
                d = get_milestones(image)
                if d is None: continue
                dn = normalisation(d)
                labels.append([emotion])
                data.append(dn.flatten())
                print (emotion)
    return labels, data



def pca (arr, keep_variance = 0.95):
    pca = sklearn.decomposition.PCA(n_components=keep_variance, svd_solver='full')
    pca.fit(arr)
    return pca

# TODO: Выделение главных компонент
