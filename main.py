#!/usr/bin/env python

# -*- coding: utf-8 -*-


from PySide.QtCore import *
from PySide.QtGui import *
import cv2
import sys
import numpy as np
import dlib
from PIL import Image,ImageDraw


# Points:   36 - left eye (left)
#           39 - left eye (right)
#           42 - right eye (left)
#           45 - right eye (right)

class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        cascadePath = "haarcascade_frontalface_alt.xml"
        self.faceCascade = cv2.CascadeClassifier(cascadePath)
        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.pose_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.setup_camera()
        self.video_size = QSize(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH), self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.latest_tilts = [0]*5
        self.setup_ui()
        self.frames= 0
        self.grey_frames = 0

    def get_faces(self, color_image):
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        # нормализуем гистограмму, и ищем лицо уже там
        # http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html
        gray_image = self.clahe.apply(gray_image)
        faces = self.frontal_face_detector(gray_image) # Лица лучше ищутся на изображениях в оттенках серого
        result = []
        for face in faces:
            current_face = {}
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            current_face['face'] = (x,y,w,h)
            shape = self.pose_model (gray_image, face)
            current_face['shape'] = shape
            result.append(current_face)
        return result


    def drawLandmarks (self, faces, image, number = False):
        for face in faces:
            (fx,fy,fw,fh) = face['face']
            cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh),
                          color=(0, 255, 0), thickness=3)
            if ('face1' in face):
                (fx1, fy1, fw1, fh1) = face['face1']
                cv2.rectangle(image, (fx1, fy1), (fx1 + fw1, fy1 + fh1),
                          color=(0, 255, 255), thickness=3)
            for i in range(0, 68):
                part = face['shape'].part(i)
                cv2.circle(image, (part.x, part.y), 2, color=(255, 255, 0), thickness = -1)
                if (number):
                    cv2.putText(image, str(i),  (part.x, part.y), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255))
        return image


    def computeTilt(self, faces):
        if (not faces): return (None, None)
        face = faces[0]['shape']
        tilt = 180 - np.arctan2(face.part(45).y - face.part(39).y, face.part(39).x - face.part(45).x)*180/np.pi
        return (tilt if tilt<180 else tilt - 360,
                (face.part(39).x + face.part(45).x / 2, face.part(39).y + face.part(45).y / 2))

    def setup_ui(self):
        """Initialize widgets.
        """
        self.image_label = QLabel()
        #self.image_label.setFixedSize(self.video_size)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.emotion_label = QLabel()
        self.emotion_label.setText("Emotion: <div style='font-size:100px;'>😐</div>")
        self.emotion_label.setToolTip("Indifferent")
        self.emotion_label.setAlignment(Qt.AlignCenter)

        self.tilt_control = QCheckBox()
        self.tilt_control.setText("Rotate image")
        self.tilt_label = QLabel()
        self.tilt_label.setText("Tilt: ")
        self.grey_label = QLabel()

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.emotion_label)
        self.main_layout.addWidget(self.tilt_control)
        self.main_layout.addWidget(self.tilt_label)
        self.main_layout.addWidget(self.grey_label)
        self.main_layout.addWidget(self.quit_button)


        self.setLayout(self.main_layout)

    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(0)
    #    self.capture.set(cv2.CV_CAP_PROP_FRAME_WIDTH, self.video_size.width())
    #    self.capture.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(50)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        faces = self.get_faces(frame)
        self.frames +=1
        #print (faces)
        frame1 = self.drawLandmarks(faces, frame, False)
        # Определяем наклон лица
        tilt, center = self.computeTilt(faces)
        # Если лицо есть на фрейме
        if tilt!=None and self.tilt_control.isChecked():
            # Смещаем плавающее окно дальше
            self.latest_tilts = self.latest_tilts[1:]
            self.latest_tilts.append(tilt)
            #Находим средний наклон за последние N фреймов
            mean_tilt = sum(self.latest_tilts)/len(self.latest_tilts)
            rows,cols,_ = frame1.shape
            # поворачиваем изображение
            M = cv2.getRotationMatrix2D((cols/2,rows/2),mean_tilt,1)
            frame1 = cv2.warpAffine(frame1,M,(cols,rows))
        image = QImage(frame1, frame1.shape[1], frame1.shape[0],
                       frame1.strides[0], QImage.Format_RGB888)
        self.tilt_label.setText("tilt: {}".format(tilt))
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.grey_label.setText("Frames: {}; Grey frames: {}".format(self.frames, self.grey_frames))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())