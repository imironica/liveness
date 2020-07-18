from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import pickle
import time
import cv2
import os
import dlib

model_path = './liveness_model.h5'

#load face detector
hog_detector = dlib.get_frontal_face_detector()

print("[INFO] loading trained model liveness detector...")
model = load_model(model_path)
currentFolder = './test_net'
imagePaths = list(paths.list_images(currentFolder))

for imagePath in imagePaths:
    image = dlib.load_rgb_image(imagePath)
    dets = hog_detector(image, 1)
    if(len(dets) == 0):
        print('None face detected')
    else:
        for i, d in enumerate(dets):
            face_crop = image[d.top():d.bottom(), d.left():d.right()]
            face_crop = cv2.resize(face_crop, (64,64))
            face = img_to_array(face_crop)
            face = np.expand_dims(face, axis=0)
            preds = model.predict(face)
            predicted_class_indices=np.argmax(preds,axis=1)
            if(predicted_class_indices[0] == 1):
                print('real')
            else:
                print('fake')
            print(imagePath)