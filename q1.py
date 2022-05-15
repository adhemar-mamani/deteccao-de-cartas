from pickle import FRAME
import cv2
import numpy as np

cap = cv2.VideoCapture("q1.mp4")
template1 = cv2.imread("cartaGrande.png", cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread("cartaPequena.png", cv2.IMREAD_GRAYSCALE)
w1, h1 = template1.shape[::-1]
w2, h2 = template2.shape[::-1]

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ret:
        break

    res1 = cv2.matchTemplate(gray_frame, template1, cv2.TM_SQDIFF_NORMED)
    res2 = cv2.matchTemplate(gray_frame, template2, cv2.TM_SQDIFF_NORMED)
    loc1 = np.where(res1 <= 0.1)
    loc2 = np.where(res2 <= 0.1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for pt in zip(*loc1[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 5)
        cv2.putText(frame, 'CARTA DETECTADA', (100, 100),
                    font, 1, (3, 0, 0), 3, cv2.LINE_4)

    for pt2 in zip(*loc2[::-1]):
        cv2.rectangle(frame, pt2, (pt2[0] + w2, pt2[1] + h2), (0, 255, 0), 5)
        cv2.putText(frame, 'CARTA DETECTADA', (100, 100),
                    font, 1, (3, 0, 0), 3, cv2.LINE_4)
    newSize = cv2.resize(frame, (1280, 720))
    cv2.imshow("Frame", newSize)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
