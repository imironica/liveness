import pandas as pd
from util import readDb
import numpy as np
import cv2
from scipy.cluster.vq import *
import os

dictionarySize = 512
descriptorType = "SIFT"

print("Computing dictionarySize: {} descriptorType: {}"
      .format(dictionarySize, descriptorType))

def computeFeatures(gray, descriptorType):
    detector = cv2.xfeatures2d.SIFT_create(100)
    (kps, descs) = detector.detectAndCompute(gray, None)
    if descs is None:
        return (None, None)
    return (kps, descs.astype("float"))

dfTrain, dfTest = readDb()
dfTrain = dfTrain.iloc[0::1]
dfTest = dfTest.iloc[0::1]

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

i = 0
print("Generate dictionary ..")
xTrain = []
yTrain = []
dictionaryList = []
i = 0
step = 5
for index, row in dfTrain.iterrows():
    path = str(row['Path']).replace("db1", "db2")
    label = int(row['Label'])

    if os.path.exists(path):
        if i % step == 0:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (kps, descs) = computeFeatures(gray, descriptorType)
            if len(dictionaryList) == 0:
                dictionaryList = descs
            dictionaryList = np.vstack((dictionaryList, descs))
        i += 1

print("Perform KMEANS clustering ..")
dictionary, variance = kmeans(dictionaryList, dictionarySize, 1)


i = 0
print("Training db ..")
xTrain = []
yTrain = []
dictionaryList = []
for index, row in dfTrain.iterrows():
    path = str(row['Path']).replace("db1", "db2")
    label = int(row['Label'])

    if os.path.exists(path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (kps, descs) = computeFeatures(gray, descriptorType)
        if descs is not None and kps is not None:
            words, distance = vq(descs, dictionary)

            feature = np.zeros(dictionary.shape[0], "int32")
            for w in words:
                feature[w] += 1
            hist = feature / sum(feature)
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
        (kps, descs) = computeFeatures(gray, descriptorType)
        if descs is not None and kps is not None:
            words, distance = vq(descs, dictionary)

            feature = np.zeros(dictionary.shape[0], "int32")
            for w in words:
                feature[w] += 1
            hist = feature / sum(feature)
            xTest.append(hist)
            yTest.append(label)

            if i % 100 == 0:
                print(i)
            i += 1

featureName = "{}_{}".format(dictionarySize, descriptorType)

df = pd.DataFrame(xTrain)
df.to_csv("../features/xTrain_{}.csv".format(featureName), index=False)

df = pd.DataFrame(xTest)
df.to_csv("../features/xTest_{}.csv".format(featureName), index=False)

df = pd.DataFrame(yTrain)
df.to_csv("../features/yTrain_{}.csv".format(featureName), index=False)

df = pd.DataFrame(yTest)
df.to_csv("../features/yTest_{}.csv".format(featureName), index=False)