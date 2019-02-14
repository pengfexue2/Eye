import cv2
import dlib
import numpy as np
from PIL import Image
"""
Python眼部识别参考链接：https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv-p-1/
"""

def img_deal(input_img):
    img = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
    rows, cols, channel = img.shape

    img_new = np.zeros((rows, cols, 4), np.uint8)
    img_new[:, :, 0:3] = img[:, :, 0:3]

    img_circle = np.zeros((rows, cols, 1), np.uint8)
    img_circle[:, :, :] = 0
    img_circle = cv2.circle(img_circle, (int(cols / 2), int(rows / 2)), int(min(rows, cols) / 2), 255, -1)

    img_new[:, :, 3] = img_circle[:, :, 0]
    cv2.imwrite('src_circle.png', img_new)


#背景图片
bg = 'haoran.jpg'
img = cv2.imread(bg)
#眼睛内容图片
src_img = "qianxi.jpg"
src = cv2.imread(src_img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faces = detector(gray)

for face in faces:
    landmarks = predictor(gray, face)
    areaX1 = max(landmarks.part(37).x, landmarks.part(41).x)
    areaX2 = min(landmarks.part(38).x, landmarks.part(40).x)
    areaY1 = max(landmarks.part(37).y, landmarks.part(38).y)
    areaY2 = min(landmarks.part(41).y, landmarks.part(40).y)
    r = int(min(areaX2-areaX1,areaY2-areaY1)/2)
    d=2*r
    midx = int((areaX1 + areaX2)/2)
    midy = int((areaY1 + areaY2)/2)

    left = cv2.resize(src, (d, d), interpolation=cv2.INTER_AREA)
    cv2.imwrite("left_eye.png", left)
    img_deal("left_eye.png")
    im = Image.open(bg)
    eye = Image.open("src_circle.png")
    im.paste(eye,(midx-r,midy-r),eye)
    #im.save("temp.png")

    area2X1 = max(landmarks.part(43).x,landmarks.part(47).x)
    area2X2 = min(landmarks.part(44).x,landmarks.part(46).x)
    area2Y1 = max(landmarks.part(43).y,landmarks.part(44).y)
    area2Y2 = min(landmarks.part(47).y,landmarks.part(46).y)
    r2 = int(min(area2X2-area2X1,area2Y2-area2Y1)/2)
    d2=2*r2
    midx2 = int((area2X1 + area2X2)/2)
    midy2 = int((area2Y1 + area2Y2)/2)

    right = cv2.resize(src, (d2, d2), interpolation=cv2.INTER_AREA)
    cv2.imwrite("right_eye.png", right)
    img_deal("right_eye.png")
    #im = Image.open('temp.png')
    eye = Image.open("src_circle.png")
    im.paste(eye,(midx2-r2,midy2-r2),eye)
    im.save("result.png")
