import cv2
import pandas as pd
import scipy.misc
import os
import dlib

def readDb():
    lstFilesTrainValid = 'db/client_train_raw.txt'
    lstFilesTrainImposter = 'db/imposter_train_raw.txt'
    lstFilesTestValid = 'db/client_test_raw.txt'
    lstFilesTestImposter = 'db/imposter_test_raw.txt'

    dfTrainValid = pd.read_csv(lstFilesTrainValid, header=None, names=['Path'])
    dfTrainValid['Path'] = 'db/ClientRaw/' + dfTrainValid['Path']
    dfTrainValid['Label'] = 1
    dfTrainImposter = pd.read_csv(lstFilesTrainImposter, header=None, names=['Path'])
    dfTrainImposter['Path'] = 'db/ImposterRaw/' + dfTrainImposter['Path']
    dfTrainImposter['Label'] = 0

    dfTestValid = pd.read_csv(lstFilesTestValid, header=None, names=['Path'])
    dfTestValid['Path'] = 'db/ClientRaw/' + dfTestValid['Path']
    dfTestValid['Label'] = 1
    dfTestImposter = pd.read_csv(lstFilesTestImposter, header=None, names=['Path'])
    dfTestImposter['Path'] = 'db/ImposterRaw/' + dfTestImposter['Path']
    dfTestImposter['Label'] = 0

    dfTrain = pd.concat([dfTrainValid, dfTrainImposter])
    dfTest = pd.concat([dfTestValid, dfTestImposter])

    return dfTrain, dfTest


def readFaces(faceCascade, path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.05, 5)

    maxArea = 0
    index = 0
    maxRectangle = (0, 0, 0, 0)

    for (x, y, w, h) in faces:
        if w * h > maxArea:
            maxArea = w * h
            maxRectangle = (x, y, w, h)

        index += 1

    roi = gray[maxRectangle[1]:maxRectangle[1] + maxRectangle[3],
                maxRectangle[0]:maxRectangle[0] + maxRectangle[2]]
    return roi, maxRectangle

  
def testFaceRecognitionAlgorithm():
    print('---- Detecting faces using cv2 HAAR Cascade')
    dfTrain, dfTest = readDb()
    #detect face with pre-trained classifier
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    invalid = 0
    i = 0
    print('Verify how many faces can be extracted on training set ...')
    for index, row in dfTrain.iterrows():
        path = row['Path']
        roi, maxRectangle = readFaces(faceCascade, path)

        if maxRectangle[2] == 0 and maxRectangle[3] == 0:
            invalid += 1
        if i % 100 == 0:
            print(i)

        i += 1

    invalidTrainDb = invalid
    print('Verify how many faces can be extracted on test set ...')
    invalid = 0
    i = 0
    for index, row in dfTest.iterrows():
        path = row['Path']
        roi, maxRectangle = readFaces(faceCascade, path)

        if maxRectangle[2] == 0 and maxRectangle[3] == 0:
            invalid += 1
        if i % 100 == 0:
            print(i)

        i += 1

    print("Training db has {} unrecognized faces".format(invalidTrainDb))
    print("Test db has {} unrecognized faces".format(invalid))

def testDlibFaceDetector():
    import dlib
    dlib.DLIB_USE_CUDA = True
    print('---- Detecting faces using dlib HOG')
    # Load face dlib detector
    hog_face_detector = dlib.get_frontal_face_detector()
    dfTrain, dfTest = readDb()
    invalid = 0

    print('Verify how many faces can be extracted on training set ...')
    for index, row in dfTrain.iterrows():
        path = row['Path']
        img = dlib.load_rgb_image(path)

        dets = hog_face_detector(img,1)

        if(len(dets) == 0):
            invalid += 1

    invalidTrainDb = invalid
    print('Verify how many faces can be extracted on test set ...')
    invalid = 0

    for index, row in dfTest.iterrows():
        path = row['Path']
        img = dlib.load_rgb_image(path)

        dets = hog_face_detector(img,1)
        if(len(dets) == 0):
            invalid += 1
        

    print("Training db has {} unrecognized faces".format(invalidTrainDb))
    print("Test db has {} unrecognized faces".format(invalid))

""" def readFaces(faceCascade, path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.05, 5)

    maxArea = 0
    index = 0
    maxRectangle = (0, 0, 0, 0)

    for (x, y, w, h) in faces:

        if w * h > maxArea:
            maxArea = w * h
            maxRectangle = (x, y, w, h)

        index += 1

    roi = gray[maxRectangle[1]:maxRectangle[1] + maxRectangle[3],
                maxRectangle[0]:maxRectangle[0] + maxRectangle[2]]
    return roi, maxRectangle """

def readColorFace(faceCascade, path):

    print(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.05, 5)

    maxArea = 0
    index = 0
    maxRectangle = (0, 0, 0, 0)

    for (x, y, w, h) in faces:
        if w * h > maxArea:
            maxArea = w * h
            maxRectangle = (x, y, w, h)

        index += 1

    roi = img[maxRectangle[1]:maxRectangle[1] + maxRectangle[3],
                maxRectangle[0]:maxRectangle[0] + maxRectangle[2]]
    return roi, maxRectangle


def saveFaces():

    dfTrain, dfTest = readDb()
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if not os.path.exists('db_faces'):
        os.mkdir('db_faces')

    if not os.path.exists('db_faces/train'):
        os.mkdir('db_faces/train')
    if not os.path.exists('db_faces/train/0'):
        os.mkdir('db_faces/train/0')
    if not os.path.exists('db_faces/train/1'):
        os.mkdir('db_faces/train/1')
    if not os.path.exists('db_faces/test'):
        os.mkdir('db_faces/test')
    if not os.path.exists('db_faces/test/0'):
        os.mkdir('db_faces/test/0')
    if not os.path.exists('db_faces/test/1'):
        os.mkdir('db_faces/test/1')


    for index, row in dfTrain.iterrows():
        path = str(row['Path'])
        label = str(row['Label'])
        roi, maxRectangle = readColorFace(faceCascade, path)
        newPath = 'db_faces/train/{}/{}.jpg'.format(label, index)
        print(newPath)
        if maxRectangle[2] > 0 and maxRectangle[3] > 0:
            scipy.misc.imsave(newPath, roi)

    for index, row in dfTest.iterrows():
        path = row['Path']
        label = str(row['Label'])
        roi, maxRectangle = readColorFace(faceCascade, path)
        newPath = 'db_faces/test/{}/{}.jpg'.format(label, index)
        if maxRectangle[2] > 0 and maxRectangle[3] > 0:
            scipy.misc.imsave(newPath, roi)

def saveFaces_hogFaceDetector():
    invalid = 0
    no_faces = 0
    dfTrain, dfTest = readDb()
    hog_detector = dlib.get_frontal_face_detector()

    if not os.path.exists('db_faces'):
        os.mkdir('db_faces')

    if not os.path.exists('db_faces/train'):
        os.mkdir('db_faces/train')
    if not os.path.exists('db_faces/train/0'):
        os.mkdir('db_faces/train/0')
    if not os.path.exists('db_faces/train/1'):
        os.mkdir('db_faces/train/1')
    if not os.path.exists('db_faces/test'):
        os.mkdir('db_faces/test')
    if not os.path.exists('db_faces/test/0'):
        os.mkdir('db_faces/test/0')
    if not os.path.exists('db_faces/test/1'):
        os.mkdir('db_faces/test/1')


    for index, row in dfTrain.iterrows():
        no_faces += 1
        imagePath = str(row['Path'])
        label = str(row['Label'])
        img = dlib.load_rgb_image(imagePath)
        newPath = 'db_faces/train/{}/{}.jpg'.format(label, index)
        print(newPath)
        dets = hog_detector(img, 1)
        if(len(dets) == 0):
                invalid += 1
                print('Unrecognized face {}'.format(imagePath))
                continue
        for i, d in enumerate(dets):
                crop = img[d.top():d.bottom(), d.left():d.right()]
                cv2.imwrite(newPath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print("--- {}/{} unrecognized Train faces".format(invalid, no_faces))

    no_faces = 0
    invalid = 0
    for index, row in dfTest.iterrows():
        no_faces += 1
        imagePath = str(row['Path'])
        label = str(row['Label'])
        img = dlib.load_rgb_image(imagePath)
        newPath = 'db_faces/test/{}/{}.jpg'.format(label, index)
        print(newPath)
        dets = hog_detector(img, 1)
        if(len(dets) == 0):
                invalid += 1
                print('Unrecognized face {}'.format(imagePath))
                continue
        for i, d in enumerate(dets):
                crop = img[d.top():d.bottom(), d.left():d.right()]
                cv2.imwrite(newPath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print("--- {}/{} unrecognized Test faces".format(invalid, no_faces))