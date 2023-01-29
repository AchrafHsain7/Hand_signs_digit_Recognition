import cv2 
import numpy as np
#from PIL import Image
from helpers import load_results
from predict_NN import *

parameters = load_results()
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    raise IOError("Error with the video capture")

printed = -1
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = capture.read()
    #used to reshape the image 
        #frame = cv2.resize(frame, None, interpolation=cv2.INTER_AREA)
    #used to flip the image around the y axis else awkward
    frame = cv2.flip(frame, 1)

    #getting the numpy arra
    image = np.array(frame)
    X = cv2.resize(image, (64,64))
    X = X.reshape(1,-1).T
    #predicting the result using the trained NN
    result = predict(X, parameters)
    #print(result)

    if result[0] == 1:
        cv2.putText(frame, '0', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 0
    elif result[1] == 1:
        cv2.putText(frame, '1', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 1
    elif result[2] == 1:
        cv2.putText(frame, '2', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 2
    elif result[3] == 1:
        cv2.putText(frame, '3', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 3
    elif result[4] == 1:
        cv2.putText(frame, '4', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 4
    elif result[5] == 1:
        cv2.putText(font, '5', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 5
 

    
    


    key = cv2.waitKey(1)
    #27 for Escape
    if key == 27:
        break