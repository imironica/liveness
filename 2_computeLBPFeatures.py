import cv2
import numpy as np
import pandas as pd
from util import readDb
from skimage.feature import local_binary_pattern
import os


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
        lbp = local_binary_pattern(gray, 24, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 24 + 3),
                                 range=(0, 24 + 2))
        hist = hist / sum(hist)
        xTrain.append(hist)
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
        lbp = local_binary_pattern(gray, 24, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 24 + 3),
                                 range=(0, 24 + 2))
        hist = hist / sum(hist)
        xTest.append(hist)
        yTest.append(label)

        if i % 100 == 0:
            print(i)
        i += 1


df = pd.DataFrame(xTrain)
df.to_csv("../features/xTrain_LBP.csv", index=False)

df = pd.DataFrame(xTest)
df.to_csv("../features/xTest_LBP.csv", index=False)

df = pd.DataFrame(yTrain)
df.to_csv("../features/yTrain_LBP.csv", index=False)

df = pd.DataFrame(yTest)
df.to_csv("../features/yTest_LBP.csv", index=False)