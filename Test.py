import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Trainer/trainner.yml')

names = np.load('Trainer/names.npy', allow_pickle=True).item()

cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cam = cv2.VideoCapture(0)

if face_cascade.empty():
    print("Error: Could not load cascade classifier. Check the path.")
    exit()
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_DUPLEX
recognized_faces = {}  
frame_count_no_face = 0

while True:
    ret, im = cam.read()
    if not ret:
        print("[Error] Could not read from camera.")
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    color = (0, 0, 255)
    status = ''

    if len(faces) == 0:
        frame_count_no_face += 1
        if frame_count_no_face > 5:
            recognized_faces.clear() 
    else:
        frame_count_no_face = 0

    for (x, y, w, h) in faces:
        face_id = (x, y, w, h)
        
        if face_id not in recognized_faces:  
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                recognized_faces[face_id] = {
                    "name": names.get(Id, "Unknown"),
                    "id": Id,
                    "conf": conf
                }
                color = (0, 255, 0)
                status = "ID Match"
            else:
                recognized_faces[face_id] = {"name": "Unknown", "id": "Unknown", "conf": conf}

        name = recognized_faces[face_id]["name"]
        Id = recognized_faces[face_id]["id"]
        color = (0, 255, 0) if recognized_faces[face_id]["conf"] < 70 else (0, 0, 255)

        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
        cv2.putText(im, f"Name: {name}", (x - 1, y - 1), font, 1, (0, 255, 0), 2)
        cv2.putText(im, f"ID: {Id}", (x - 1, y + 20), font, 1, (0, 255, 0), 2)

    cv2.putText(im, status, (100, 100), font2, 2, color, 2)
    cv2.imshow('Face Recognition', im)

    if cv2.waitKey(10) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


