import sys
import os
import cv2
from skimage import io
import dlib
from imutils import paths
from utils import image_resize

#HOG face detector
hog_detector = dlib.get_frontal_face_detector()

dataset_path = './dataset/{}/'
folders = ['real','fake']

faces_path = './face_dataset/'

data = []
labels = []

dlib.DLIB_USE_CUDA = True

def extract_faces():
    for folder in folders:
        currentFolder = dataset_path.format(folder)
        imagePaths = list(paths.list_images(currentFolder))
        invalid = 0
        no_faces = 0
        print('--- Detecting {} faces...'.format(folder))
        for imagePath in imagePaths:
            no_faces += 1
            img = dlib.load_rgb_image(imagePath)
            img_name = imagePath.split('/')[-1]
            image = image_resize(img, height=500)
            dets = hog_detector(image, 1)
            
            if(len(dets) == 0):
                invalid += 1
                print('Unrecognized face {}'.format(imagePath))
                continue
            
            for i, d in enumerate(dets):
                crop = image[d.top():d.bottom(), d.left():d.right()]
                path_to_save_faces = faces_path + folder + '/'
                if not os.path.exists(faces_path):
                    os.mkdir(faces_path)
                if not os.path.exists(path_to_save_faces):
                    os.mkdir(path_to_save_faces)
                cv2.imwrite(path_to_save_faces + img_name, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        print("--- {}/{} unrecognized {} faces".format(invalid, no_faces, folder))
