import cv2
import pandas as pd
import os
from skimage.feature import hog

folderImages = 'db_faces/{}/'

i = 0
print("Training db ..")
xTrain = []
yTrain = []


currentFolder = folderImages.format('train')
print('Compute train features ')

folders = ['0', '1']
for folder in folders:
    images = [os.path.join(currentFolder, folder,  file) for file in os.listdir(currentFolder + folder)]

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = hog(gray, orientations=8)
        xTrain.append(feats)
        yTrain.append(int(folder))

        if i % 100 == 0:
            print(i)
        i += 1


xTest = []
yTest = []
i = 0

currentFolder = folderImages.format('test')
print('Compute test features ')

folders = ['0', '1']
for folder in folders:
    folder2 = os.path.join(currentFolder, folder)
    print(folder2)
    images = [os.path.join(currentFolder, folder,  file) for file in os.listdir(os.path.join(currentFolder, folder))]

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = hog(gray, orientations=8)
        xTest.append(feats)
        yTest.append(int(folder))

        if i % 100 == 0:
            print(i)
        i += 1


if not os.path.exists('features'):
    os.mkdir('features')

df = pd.DataFrame(xTrain)
df.to_csv("features/xTrain_HOG.csv", index=False)

df = pd.DataFrame(xTest)
df.to_csv("features/xTest_HOG.csv", index=False)

df = pd.DataFrame(yTrain)
df.to_csv("features/yTrain_HOG.csv", index=False)

df = pd.DataFrame(yTest)
df.to_csv("features/yTest_HOG.csv", index=False)



