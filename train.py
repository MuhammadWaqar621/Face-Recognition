from imutils import paths
import imutils
import face_recognition
import cv2
import os
import pickle
import time
from pathlib import Path
Path("models").mkdir(parents=True, exist_ok=True)
print('[INFO] creating facial embeddings...')

ti = time.time()
knownEncodings, knownNames = [], []
imagePaths = list(paths.list_images(
    os.getcwd() + '\\100 faces each\\Captured images'))  # dataset here
for (i, imagePath) in enumerate(imagePaths):
    print('{}/{}'.format(i+1, len(imagePaths)), end=', ')
    image, name = cv2.imread(imagePath), imagePath.split(os.path.sep)[-2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(
        rgb,  model='cnn')  # detection_method here
    for encoding in face_recognition.face_encodings(rgb, boxes, model='large'):
        knownEncodings.append(encoding)
        knownNames.append(name)
data = {'encodings': knownEncodings, 'names': knownNames}
f = open(os.getcwd() + '\\models\\100_encodings.pickle', 'wb')
f.write(pickle.dumps(data))
f.close()
print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))
