import cv2
import numpy as np
import pandas as pd
from util import readDb
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import os
import mahotas.features
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
from skimage.feature import local_binary_pattern

dfTrain, dfTest = readDb()
dfTrain = dfTrain.iloc[0::1]
dfTest = dfTest.iloc[0::1]

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

i = 0
print("Training db ..")
xTrain = []
yTrain = []

for index, row in dfTrain.iterrows():
    path = str(row['Path']).replace("db1", "db2")
    label = int(row['Label'])

    if os.path.exists(path):
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
        xTrain.append(feats)
        yTrain.append(label)

        if i % 100 == 0:
            print(i)
        i += 1

xTest = []
yTest = []
i = 0
for index, row in dfTest.iterrows():
    path = str(row['Path']).replace("db1", "db2")
    label = int(row['Label'])

    if os.path.exists(path):
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

        xTest.append(feats)
        yTest.append(label)

        if i % 100 == 0:
            print(i)
        i += 1


df = pd.DataFrame(xTrain)
df.to_csv("../features/xTrain_glcm.csv", index=False)

df = pd.DataFrame(xTest)
df.to_csv("../features/xTest_glcm.csv", index=False)

df = pd.DataFrame(yTrain)
df.to_csv("../features/yTrain_glcm.csv", index=False)

df = pd.DataFrame(yTest)
df.to_csv("../features/yTest_glcm.csv", index=False)