from cv2 import cv2
import time
import pandas
from datetime import datetime as dt

video = cv2.VideoCapture(0)
frame_1 = None
status = [None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

while (True):
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Grayscale", gray)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    cv2.imshow("Blur", gray)
    state = 0

    if frame_1 is None:
        frame_1 = gray
        continue
    delta_frame = cv2.absdiff(gray, frame_1)

    cv2.imshow("Delta", delta_frame)

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)

    (cnts, _) = cv2.findContours(thresh_frame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 2000:
            continue
        state = 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    status.append(state)

    if status[-1] == 1 and status[-2] == 0:
        times.append(dt.now())
    elif status[-1] == 0 and status[-2] == 1:
        times.append(dt.now())

    cv2.imshow("Initial Frame", frame_1)
    cv2.imshow("Threshold", thresh_frame)
    cv2.imshow("Rectangles", frame)

    if cv2.waitKey(30) == ord('p'):
        if state == 1:
            times.append(dt.now())
        break

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)

print(df)

video.release()
cv2.destroyAllWindows()
