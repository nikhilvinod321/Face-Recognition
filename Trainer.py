import cv2
import os
import numpy as np
from PIL import Image
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  
    faceSamples = []
    Ids = []
    name_map = {}

    for imagePath in imagePaths:
        if os.path.split(imagePath)[-1].split(".")[-1] != 'jpg':
            continue
        pilImage = Image.open(imagePath).convert('L')
        print(f"Processing {imagePath}...") 
        imageNp = np.array(pilImage, 'uint8')
        
        filename = os.path.split(imagePath)[-1]
        parts = filename.split("_")
        
        if len(parts) >= 2:
            user_name = parts[0] 
            user_id = int(parts[1].replace("ID", "")) 
            name_map[user_id] = user_name  
        
        print(f"User Name: {user_name}, ID: {user_id}")
        faces = detector.detectMultiScale(imageNp)
        print('Extracting Face...') 
        for (x, y, w, h) in faces:
            print(f'Adding cropped image of {user_name} (ID: {user_id}) to face sample archive...')
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(user_id)

    return faceSamples, Ids, name_map

faces, Ids, name_map = getImagesAndLabels('Images')
print('[+] Analysis in progress...')
recognizer.train(faces, np.array(Ids))
recognizer.save('Trainer/trainner.yml')
print('[!!!] Image Analysis Complete!')

os.makedirs('Trainer', exist_ok=True)
np.save('Trainer/names.npy', name_map)
print("names.npy file saved successfully with ID-to-name mappings.")

time.sleep(2)
