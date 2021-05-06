import cv2
from random import randrange


trained_face_data = cv2.CascadeClassifier('Faces.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for face in face_coordinates:
        cv2.rectangle(frame, face, (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('Show!', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break
