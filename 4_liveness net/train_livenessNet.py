# import the necessary packages
from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
from keras import backend as K
import matplotlib.pyplot as plt
import dlib
import numpy as np
import argparse
import cv2
import os

# dataset_path = '../db_faces/{}/'

model_path = 'liveness_model.h5'

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 32
EPOCHS = 100

# construct the training image generator for data augmentation
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory('../db_faces/train',
                                                 target_size=(64,64),
                                                 color_mode='rgb',
                                                 batch_size=BS,
                                                 class_mode='categorical',
                                                 shuffle=False)

validation_generator=train_datagen.flow_from_directory('../db_faces/test',
                                                 target_size=(64,64),
                                                 color_mode='rgb',
                                                 batch_size=BS,
                                                 class_mode='categorical',
                                                 shuffle=False)
 
labels = (train_generator.class_indices)
# initialize the optimizer and model
adam_opt = Adam(lr=INIT_LR, decay= 1e-5)
model = LivenessNet.build(width=64, height=64, depth=3,
	classes=len(labels))

for layer in model.layers:
    print(layer.output_shape)

print("[INFO] compiling model...")
#configure the learning process
model.compile(loss="binary_crossentropy", optimizer=adam_opt,
	metrics=["accuracy"])

step_size_train = train_generator.n//train_generator.batch_size
step_size_validation = validation_generator.samples // validation_generator.batch_size

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = validation_generator,
                   validation_steps = step_size_validation,
                   epochs=EPOCHS,
				   shuffle=False,
				    )

# save the network to disk
print("[INFO] serializing network to '{}'...".format(model_path))
model.save(model_path)

print("[INFO] Class indices")
labels = (train_generator.class_indices)
print(labels)

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