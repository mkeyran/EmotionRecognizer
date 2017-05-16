#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    pyside = False
    from PyQt5.QtCore import QSize, QRect, Qt, QTimer, QRectF
    from PyQt5.QtGui import QPainter, QImage, QPixmap
    from PyQt5.QtWidgets import QWidget, QLabel, QCheckBox, QPushButton
    from PyQt5.QtWidgets import QVBoxLayout, QApplication
    from PyQt5 import QtSvg
except:
    print ("Reverting to PySide")
    pyside = True
    from PySide.QtCore import QSize, QRect, Qt, QTimer
    from PySide.QtGui import QWidget, QPainter, QLabel, QCheckBox, QPushButton
    from PySide.QtGui import QVBoxLayout, QImage, QPixmap, QApplication
    from PySide import QtSvg

import ctypes
import cv2
import sys
import numpy as np
import preprocess
import nn_learn
import pickle
import mappings
import decision_trees_ensemble

from pympler.tracker import SummaryTracker

emotions_labels = ["Neutral", "Anger", "Contempt",
                   "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
emotions_smileys = ["😐", "😠", "😒", "😖", "😱", "😄", "😭", "😲"]
# Emojis from http://emojione.com/developers/
emotions_svg = ["1f610", "1f621", "1f612",
                "1f616", "1f631", "1f603", "1f61f", "1f628"]
svg_path = "Emotions/svg"
# Points:   36 - left eye (left)
#           39 - left eye (right)
#           42 - right eye (left)
#           45 - right eye (right)

emotions_moving_average_N = 20

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]



class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setup_camera()
        self.video_size = QSize(self.capture.get(
            cv2.CAP_PROP_FRAME_WIDTH), self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.latest_tilts = [0] * 5
        self.setup_ui()
        self.frames = 0
        self.grey_frames = 0
        self.pca = pickle.load(open("data/TrainingData/pcamapping.dat",'rb'))
        self.last_emotions = []
         #import pdb;        pdb.set_trace();

        self.maps = (mappings.DropContemptMapping(),
                         mappings.NormalizeMapping(), 
                         mappings.ImageMirrorMapping(),
                         self.pca
                         )
        self.modelName = "model"
        self.model = nn_learn.NeuralNetwork(nn_learn.neural_net2_pca)
        self.model.load()
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(50)


    def recognize_emotion(self, faces):
        if not faces.faces:
            return []
        faces.emotions = np.array(self.model.predict(faces.generate_data()))
        # Наивное применение движущегося среднего
        # TODO: Добавить определение среднего для нескольких лиц
        emotions = faces.emotions
        for le in self.last_emotions:
            if le.emotions is not None:
                emotions = emotions + le.emotions
        return np.argmax(faces.emotions, axis = 1).tolist()
 

    def drawLandmarks(self, faces, image, number=False, frames=False, dots=False):
        for face in faces.faces:
            if frames:
                (fx, fy, fw, fh) = (face.rectangle.left(), face.rectangle.top(),
                                    face.rectangle.width(), face.rectangle.height())
                cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh),
                              color=(0, 255, 0), thickness=3)
            for i in range(0, 68):
                part = face.milestones()[i]
                if dots:
                    cv2.circle(image, (int(part[0]), int(part[1])), 2, color=(
                        255, 255, 0), thickness=-1)
                if (number):
                    cv2.putText(image, str(i),  (int(part[0]), int(
                        part[1])), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255))
        return image

    def drawSmileys(self, faces, emotions, image):
        painter = QPainter()
        painter.begin(image)
        for i, face in enumerate(faces.faces):
            drect = face.rectangle
            rect = QRectF(drect.left(), drect.top(),
                         drect.width(), drect.height())
            painter.setRenderHints(QPainter.Antialiasing, True)
            svg = QtSvg.QSvgRenderer()
            svg.load(svg_path + "/" + emotions_svg[emotions[i]] + ".svg")
            svg.render(painter,  rect)
        painter.end()
        return image

    def setup_ui(self):
        """Initialize widgets.
        """
        self.image_label = QLabel()
        # self.image_label.setFixedSize(self.video_size)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.emotion_label = QLabel()
        self.emotion_label.setText(
            u"Emotion: <div style='font-size:100px;'>😐</div>")
        self.emotion_label.setToolTip("Indifferent")
        self.emotion_label.setAlignment(Qt.AlignCenter)

        self.tilt_control = QCheckBox()
        self.tilt_control.setText("Rotate image")
        self.tilt_label = QLabel()
        self.tilt_label.setText("Tilt: ")
        self.grey_label = QLabel()

        self.anonymousMode = QCheckBox()
        self.anonymousMode.setText("Anonymous mode")

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.emotion_label)
        self.main_layout.addWidget(self.tilt_control)
        self.main_layout.addWidget(self.tilt_label)
        self.main_layout.addWidget(self.grey_label)
        self.main_layout.addWidget(self.anonymousMode)
        self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)

    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(0)
        #    self.capture.set(cv2.CV_CAP_PROP_FRAME_WIDTH, self.video_size.width())
        #    self.capture.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, self.video_size.height())


    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        #frame = cv2.blur(frame, (2 ,2))
        #faces = self.get_faces(frame)
        faces = preprocess.FaceSet(preprocess.Face.fabric(image=frame))
        faces.mappings = self.maps
        self.frames += 1
        #if (self.frames == 10):
        #    self.tracker = SummaryTracker()

        #if (self.frames == 100):
        #    self.tracker.print_diff()
        #if (self.frames == 200):
        #    self.timer.stop()
        #    import pdb; pdb.set_trace()
        
        #print (faces)
        emotions = self.recognize_emotion(faces)
        emotion_text = "Emotion:"
        for emotion in emotions:
            emotion_text += "<div style='font-size:100px;'>{}</div>".format(
                emotions_smileys[emotion])
        frame1 = self.drawLandmarks(faces, frame, False)
        self.emotion_label.setText(emotion_text)
        # Определяем наклон лица
        if faces.faces:
            tilt = faces.faces[0].tilt()
        else:
            tilt = None
        # Если лицо есть на фрейме
        if tilt != None and self.tilt_control.isChecked():
            # Смещаем плавающее окно дальше
            self.latest_tilts = self.latest_tilts[1:]
            self.latest_tilts.append(tilt)
            # Находим средний наклон за последние N фреймов
            mean_tilt = sum(self.latest_tilts) / len(self.latest_tilts)
            rows, cols, _ = frame1.shape
            # поворачиваем изображение
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), mean_tilt, 1)
            frame1 = cv2.warpAffine(frame1, M, (cols, rows))
        rcount = ctypes.c_long.from_address(id(frame1)).value # Get the reference count of self.data.

        image = QImage(frame1, frame1.shape[1], frame1.shape[0],
                       frame1.strides[0], QImage.Format_RGB888)

        ## Обход утечки памяти из-за QImage в Pyside
        if pyside:
            ctypes.c_long.from_address(id(frame1)).value=rcount

        if self.anonymousMode.isChecked():
            image = self.drawSmileys(faces, emotions, image)

        self.tilt_label.setText("tilt: {}".format(tilt))
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.grey_label.setText("Frames: {}; Grey frames: {}".format(
            self.frames, self.grey_frames))
        
        # Добавляем эмоцию к эмоциям, применяющимся для движущегося среднего
        if (len(self.last_emotions) >= emotions_moving_average_N):
            self.last_emotions.pop()
        self.last_emotions.append (faces)

    def __del__(self):
        self.capture.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
