import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam

INIT_LR = 1e-4
BS = 32
EPOCHS = 10

# MobileNetV2
base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=2)

x=base_model.output # a luat outputul inainte de FC
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

# specify the inputs
# specify the outputs
# now a model has been created based on our architecture
new_model=Model(inputs=base_model.input, outputs=preds)

# check the model architecture
for i,layer in enumerate(new_model.layers):
  print(i,layer.name)

# We are not using weights pre-trained on ImageNet.
for layer in new_model.layers:
    layer.trainable=True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.25)

train_generator=train_datagen.flow_from_directory('E:/ReactApp/liveness/liveness/4_liveness net/face_dataset',
                                                 target_size=(64,64),
                                                 color_mode='rgb',
                                                 batch_size=BS,
                                                 class_mode='categorical',
                                                 shuffle=True)

validation_generator=train_datagen.flow_from_directory('E:/ReactApp/liveness/liveness/4_liveness net/face_dataset',
                                                 target_size=(64,64),
                                                 color_mode='rgb',
                                                 batch_size=BS,
                                                 class_mode='categorical',
                                                 shuffle=True)

# Adam optimizer
adam_opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# loss function will be binary cross entropy
# evaluation metric will be accuracy
new_model.compile(optimizer=adam_opt, loss="binary_crossentropy", metrics=['accuracy'])

step_size_train = train_generator.n//train_generator.batch_size
step_size_validation = validation_generator.samples // validation_generator.batch_size
H = new_model.fit_generator(train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = validation_generator,
                   validation_steps = step_size_validation,
                   epochs=EPOCHS)

# save the network to disk
print("[INFO] serializing network and save it to disk...")
new_model.save('liveness.model')

print("[INFO] Class indices")
labels = (train_generator.class_indices)
print(labels)