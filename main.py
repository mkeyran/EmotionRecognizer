#!/usr/bin/env python
# -*- coding: utf-8 -*-


from PySide.QtCore import *
from PySide.QtGui import *
import PySide.QtSvg
import cv2
import sys
import numpy as np
import dlib
from PIL import Image,ImageDraw
import preprocess
import nn_learn

emotions_labels = ["Neutral", "Anger", "Contempt", "Disgust","Fear","Happiness","Sadness","Surprise"]
emotions_smileys = ["😐","😠","😒","😖","😱","😄","😭","😲"]
# Emojis from http://emojione.com/developers/
emotions_svg=["1f610","1f621","1f612","1f616","1f631","1f603", "1f61f","1f628"]
svg_path = "Emotions/svg"
# Points:   36 - left eye (left)
#           39 - left eye (right)
#           42 - right eye (left)
#           45 - right eye (right)

class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setup_camera()
        self.video_size = QSize(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH), self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.latest_tilts = [0]*5
        self.setup_ui()
        self.frames= 0
        self.grey_frames = 0
        self.modelName = "model"
        self.model = nn_learn.get_network()
        self.model.load (self.modelName)


    def recognize_emotion(self, faces):
        emotions = []
        for face in faces:
            predicted = np.array(self.model.predict(nn_learn.pca.transform(preprocess.normalisation(face[1]).flatten())))
            emotions.append(predicted.argmax())
        return emotions

    def drawLandmarks (self, faces, image, number = False, frames= False, dots = False):
        for face in faces:
            if frames:
                (fx,fy,fw,fh) = (face[0].left(), face[0].top(), face[0].width(), face[0].height())
                cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh),
                          color=(0, 255, 0), thickness=3)
            for i in range(0, 68):
                part = face[1][i]
                if dots:
                    cv2.circle(image, (int (part[0]), int(part[1])), 2, color=(255, 255, 0), thickness = -1)
                if (number):
                    cv2.putText(image, str(i),  (int (part[0]), int(part[1])), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255))
        return image

    def drawSmileys(self, faces, emotions, image):
        painter = QPainter()
        painter.begin(image)
        for i, face in enumerate (faces):
            rect = QRect (face[0].left(), face[0].top(), face[0].width(), face[0].height())
            painter.setRenderHints(QPainter.Antialiasing, True)
            svg = PySide.QtSvg.QSvgRenderer()
            svg.load(svg_path+"/"+emotions_svg[emotions[i]]+".svg")
            svg.render(painter, rect)
        painter.end()
        return image

        

    def computeTilt(self, faces):
        if (not faces): return (None, None)
        face = faces[0][1]
        tilt = 180 - np.arctan2(face[45][1]-face[39][1], face[39][0]-face[45][0])*180/np.pi
        return (tilt if tilt<180 else tilt - 360,
                (face[39][0] + face[45][0] / 2, face[39][1] + face[45][1] / 2))

    def setup_ui(self):
        """Initialize widgets.
        """
        self.image_label = QLabel()
        #self.image_label.setFixedSize(self.video_size)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.emotion_label = QLabel()
        self.emotion_label.setText(u"Emotion: <div style='font-size:100px;'>😐</div>")
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

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(50)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        #faces = self.get_faces(frame)
        faces = preprocess.get_milestones(frame)
        self.frames +=1
        #print (faces)

        emotions = self.recognize_emotion(faces)
        emotion_text= "Emotion:"
        for emotion in emotions:
            emotion_text +=  "<div style='font-size:100px;'>{}</div>".format(emotions_smileys[emotion])
        frame1 = self.drawLandmarks(faces, frame, False)
        self.emotion_label.setText(emotion_text)
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
        if self.anonymousMode.isChecked(): image = self.drawSmileys(faces, emotions, image)

        self.tilt_label.setText("tilt: {}".format(tilt))
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.grey_label.setText("Frames: {}; Grey frames: {}".format(self.frames, self.grey_frames))
    
    def __del__(self):
        self.capture.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())