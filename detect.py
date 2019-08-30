from cv2 import cv2
import time
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_1 = None

while (True):
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if frame_1 is None:
        frame_1 = gray
        continue
    delta_frame = cv2.absdiff(gray, frame_1)

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)
    
    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    
    cv2.imshow("Gray", frame_1)
    cv2.imshow("Delta", delta_frame)
    cv2.imshow("Threshold", thresh_frame)
    cv2.imshow("Rectangles", frame)
    
    
    #print(delta_frame)

    if cv2.waitKey(1) == ord('a'):
        break

video.release()
cv2.destroyAllWindows()