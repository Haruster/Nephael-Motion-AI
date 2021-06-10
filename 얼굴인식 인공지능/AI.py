import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('/Users/jeongjun-young/Desktop/얼굴인식 인공지능/PHOTO.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade_file = '/usr/local/Cellar/opencv/4.5.2_4/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(cascade_file)
face = face_cascade.detectMultiScale(gray, 1.2, 5)

print("Number of faces detected: " + str(len(face)))

if len(face):
	for (x, y, w, h) in face:
		cv2.rectangle(image, (x, y), (x + y, y + h), (255, 0, 0), 8)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=gray)
plt.xticks([]), plt.yticks([])
plt.show
