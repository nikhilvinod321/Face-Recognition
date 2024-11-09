import numpy as np
import cv2
import os

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0) 

if detector.empty():
    print("Error: Could not load the cascade classifier.")
    exit()
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

user_name = input("Enter the Name: ")
user_id = input("Enter the ID #: ")
output_dir = "Images"
os.makedirs(output_dir, exist_ok=True)

sample_num = 1
while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Could not capture image from camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sample_num += 1
        face_img = gray[y:y+h, x:x+w]
        file_path = os.path.join(output_dir, f"{user_name}_ID{user_id}_{sample_num}.jpg")
        cv2.imwrite(file_path, face_img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
        cv2.putText(img, f"{user_name} ID:{user_id}", (x - 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video Feed - Press Q to Stop', img)
    if sample_num > 210:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Collection complete!!')
