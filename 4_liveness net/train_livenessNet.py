# import the necessary packages
from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
from extract_faces import extract_faces
import matplotlib.pyplot as plt
import dlib
import numpy as np
import argparse
import pickle
import cv2
import os

dataset_path = './face_dataset'
model_path = './liveness.model'
label_encoder = './le.pickle'

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 64
EPOCHS = 50

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading face images...")
detector = dlib.get_frontal_face_detector()
imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []
 
for imagePath in imagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be 64x64 pixels
	label = imagePath.split(os.path.sep)[-2]
	
	image = dlib.load_rgb_image(imagePath)
	
	image = cv2.resize(image, (64, 64))
	
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data into a NumPy array
data = np.array(data, dtype="float")

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# construct the training image generator for data augmentation
augment = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
 
# initialize the optimizer and model
print("[INFO] compiling model...")
adam_opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=64, height=64, depth=3,
	classes=len(le.classes_))
for layer in model.layers:
    print(layer.output_shape)

#configure the learning process
model.compile(loss="binary_crossentropy", optimizer=adam_opt,
	metrics=["accuracy"])
 
# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(augment.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# save the network to disk
print("[INFO] serializing network to '{}'...".format(model_path))
model.save(model_path)

# save the label encoder to disk
f = open(label_encoder, "wb")
f.write(pickle.dumps(le))
f.close()

# summarize history for accuracy
plt.plot(H.history['acc'])
plt.plot(H.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()