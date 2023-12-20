import cv2
import cvzone
import numpy as np


def nothing(a):
    pass


cv2.namedWindow("image")
cv2.resizeWindow("image", 640, 470)
cv2.createTrackbar('thresh1', 'image', 255, 255, nothing)
cv2.createTrackbar('thresh2', 'image', 255, 255, nothing)
totalMoney = 0


def processImg(imgPre):
    imgPre = cv2.GaussianBlur(imgPre, (5, 5), 1)
    thersh1 = cv2.getTrackbarPos("thresh1", "image")
    thersh2 = cv2.getTrackbarPos("thresh2", "image")

    imgPre = cv2.Canny(imgPre, thersh1, thersh2)
    kernel = np.ones((1, 1), np.uint8)
#     img = cv2.dilate(img, (5, 5), iterations=15)
    imgPre = cv2.erode(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)
    return imgPre


while True:
    totalMoney = 0
    img = cv2.imread("images/img1.png")
    img = cv2.resize(img, (640, 470), None)
    imgPre = processImg(img)
    imgContors, conFind = cvzone.findContours(img, imgPre, minArea=20)

    for contour in conFind:
        peri = cv2.arcLength(contour["cnt"], True)
        approx = cv2.approxPolyDP(contour["cnt"], 0.02 * peri, True)
        # print(len(approx))
        if len(approx) > 5:
            area = contour["area"]
            if area < 2050:
                totalMoney += 5
            elif 2050 < area < 2500:
                totalMoney += 1
            else:
                totalMoney += 2
    print(totalMoney)

    cvzone.putTextRect(img, f'{totalMoney} $', (50, 50), colorT=(0, 255, 0))

    AllImg = cvzone.stackImages([img, imgPre, imgContors], 2, 1)
    cv2.imshow("img", AllImg)

#     cv2.imshow("processImg", processImg)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
