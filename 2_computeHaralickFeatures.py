import cv2
import pandas as pd
import os
import numpy as np
from skimage.feature import greycomatrix, greycoprops


def computeHaralick(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    properties = ['energy', 'homogeneity']

    glcm = greycomatrix(gray,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)

    feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
    return feats

folderImages = 'db_faces/{}/'


print("Training db ..")
xTrain = []
yTrain = []


currentFolder = folderImages.format('train')
print('Compute train features ')

folders = ['0', '1']
for folder in folders:
    print('Processing folder ' + folder)
    images = [os.path.join(currentFolder, folder,  file) for file in os.listdir(currentFolder + folder)]
    i = 0
    for path in images:
        feats = computeHaralick(path)
        xTrain.append(feats)
        yTrain.append(int(folder))

        if i % 100 == 0:
            print(i)
        i += 1


xTest = []
yTest = []


currentFolder = folderImages.format('test')
print('Compute test features ')

folders = ['0', '1']
for folder in folders:

    print('Processing folder ' + folder)

    images = [os.path.join(currentFolder, folder,  file) for file in os.listdir(os.path.join(currentFolder, folder))]
    i = 0
    for path in images:
        feats = computeHaralick(path)
        xTest.append(feats)
        yTest.append(int(folder))

        if i % 100 == 0:
            print(i)
        i += 1


if not os.path.exists('features'):
    os.mkdir('features')

df = pd.DataFrame(xTrain)
df.to_csv("features/xTrain_glcm.csv", index=False)

df = pd.DataFrame(xTest)
df.to_csv("features/xTest_glcm.csv", index=False)

df = pd.DataFrame(yTrain)
df.to_csv("features/yTrain_glcm.csv", index=False)

df = pd.DataFrame(yTest)
df.to_csv("features/yTest_glcm.csv", index=False)