import cv2
import numpy as np

frame_width = 960
frame_height = 540
task_video = cv2.VideoCapture('Test task1_video.mp4')
task_video.set(3, frame_width)
task_video.set(4, frame_height)


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 960, 270)
cv2.createTrackbar("Threshold1", "Parameters", 20, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 20, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)


def stack_images(scale,img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape [:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        imageblank = np.zeros((height, width, 3), np.uint8)
        hor = [imageblank]*rows
        hor_con = [imageblank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None,scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def get_contours(img, contour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            shape_type = ''
            shape_color = (255, 255, 255)
            if len(approx) == 3:
                shape_type = "Triangle"
                shape_color = (255, 0, 0)
            elif len(approx) == 4:
                shape_type = "Rectangle"
                shape_color = (0, 0, 255)
            elif len(approx) == 5:
                shape_type = "Pentagon"
                shape_color = (75, 0, 130)
            else:
                shape_type = "Circle"
                shape_color = (255, 0, 255)
            cv2.drawContours(contour, cnt, -1, shape_color, 4)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(contour, (x, y), (x + w, y + h), (0, 255, 102), 2)

            cv2.putText(contour, shape_type, (x + w + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (0, 255, 102), 2)
            cv2.putText(contour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 102), 2)


while True:
    success, fr0 = task_video.read()
    table = np.array([2*i-255 if i > 127 else 0
                      for i in np.arange(0, 256)]).astype("uint8")
    fr = cv2.LUT(fr0, table)
    contour = fr.copy()
    blurred = cv2.GaussianBlur(fr, (7, 7), 1)
    togray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    canny_detect = cv2.Canny(togray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(canny_detect, kernel, iterations=1)
    get_contours(dilation, contour)
    img_stack = stack_images(0.8, ([fr0, fr],
                                  [dilation, contour]))
    cv2.imshow("Result", img_stack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
