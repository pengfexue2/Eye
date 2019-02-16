import cv2
import math
import dlib
from PIL import Image
import numpy as np



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

cap = cv2.VideoCapture(0)



# #眼睛内容图片
# src_img = "mia1.jpg"
# src = cv2.imread(src_img)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#眼睛内容图片
src_img = "eye.png"
src = cv2.imread(src_img)

while True:
    _, frame = cap.read()
    bg = "temp.jpg"
    cv2.imwrite(bg,frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x,y = face.left(),face.top()
        x1,y1 = face.right(),face.bottom()
        #cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        landmarks = predictor(gray,face)
        left_top = (landmarks.part(37).x,landmarks.part(37).y)
        right_top = (landmarks.part(38).x, landmarks.part(38).y)
        left_bot = (landmarks.part(41).x, landmarks.part(41).y)
        right_bot = (landmarks.part(40).x, landmarks.part(40).y)

        # l1 = cv2.line(frame,left_top,right_top,(0,255,0),2)
        # l2 = cv2.line(frame,right_top,right_bot,(0, 255, 0), 2)
        # l3 = cv2.line(frame, left_bot, right_bot, (0, 255, 0), 2)
        # l4 = cv2.line(frame, left_top, left_bot, (0, 255, 0), 2)
        areaX1 = max(landmarks.part(37).x, landmarks.part(41).x)
        areaX2 = min(landmarks.part(38).x, landmarks.part(40).x)
        areaY1 = max(landmarks.part(37).y, landmarks.part(38).y)
        areaY2 = min(landmarks.part(41).y, landmarks.part(40).y)

        r = int(math.sqrt(math.pow((areaX2 - areaX1), 2) + math.pow((areaY2 - areaY1), 2)) / 2)
        # r = int(min(areaX2-areaX1,areaY2-areaY1))
        d = 2 * r
        midx = int((areaX1 + areaX2) / 2)
        midy = int((areaY1 + areaY2) / 2)

        left = cv2.resize(src, (d, d), interpolation=cv2.INTER_AREA)
        cv2.imwrite("left_eye.png", left)
        img_deal("left_eye.png")
        im = Image.open(bg)
        eye = Image.open("src_circle.png")
        im.paste(eye, (midx - r, midy - r), eye)

        left_top2 = (landmarks.part(43).x,landmarks.part(43).y)
        right_top2 = (landmarks.part(44).x, landmarks.part(44).y)
        left_bot2 = (landmarks.part(47).x, landmarks.part(47).y)
        right_bot2 = (landmarks.part(46).x, landmarks.part(46).y)

        # l12 = cv2.line(frame,left_top2,right_top2,(0,255,0),2)
        # l22 = cv2.line(frame,right_top2,right_bot2,(0, 255, 0), 2)
        # l32 = cv2.line(frame, left_bot2, right_bot2, (0, 255, 0), 2)
        # l42 = cv2.line(frame, left_top2, left_bot2, (0, 255, 0), 2)

        area2X1 = max(landmarks.part(43).x, landmarks.part(47).x)
        area2X2 = min(landmarks.part(44).x, landmarks.part(46).x)
        area2Y1 = max(landmarks.part(43).y, landmarks.part(44).y)
        area2Y2 = min(landmarks.part(47).y, landmarks.part(46).y)
        r2 = int(math.sqrt(math.pow((area2X2 - area2X1), 2) + math.pow((area2Y2 - area2Y1), 2)) / 2)
        # r2 = int(min(area2X2-area2X1,area2Y2-area2Y1)*2/3)
        d2 = 2 * r2
        midx2 = int((area2X1 + area2X2) / 2)
        midy2 = int((area2Y1 + area2Y2) / 2)

        right = cv2.resize(src, (d2, d2), interpolation=cv2.INTER_AREA)
        cv2.imwrite("right_eye.png", right)
        img_deal("right_eye.png")
        # im = Image.open('temp.png')
        eye = Image.open("src_circle.png")
        im.paste(eye, (midx2 - r2, midy2 - r2), eye)
        im.save("result.png")

    cv2.imshow("Frame", cv2.imread("result.png"))

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()