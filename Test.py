from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)       #procesarea imaginii
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur, 50, 150)
    return gray

def videocapture():                         #procesarea videoclipului
    cap = cv2.VideoCapture("test2.mp4")
    while(cap.isOpened()):
        _,frame = cap.read()
        canny_image = canny(frame)
        plt.imshow(canny_image)
        plt.show()
        #cv2.imshow("result", canny_image)
        #cv2.waitKey(1)

videocapture()