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
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

INIT_LR = 1e-5
BS = 32
EPOCHS = 20

# MobileNetV2
base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=2)

x=base_model.output # a luat outputul inainte de FC
x=GlobalAveragePooling2D()(x)
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

train_datagen=ImageDataGenerator()

train_generator=train_datagen.flow_from_directory('../db_faces/train',
                                                 target_size=(64,64),
                                                 color_mode='rgb',
                                                 batch_size=BS,
                                                 class_mode='binary',
                                                 shuffle=True)

validation_generator=train_datagen.flow_from_directory('../db_faces/test',
                                                 target_size=(64,64),
                                                 color_mode='rgb',
                                                 batch_size=BS,
                                                 class_mode='binary',
                                                 shuffle=False)

# Adam optimizer
adam_opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# loss function will be binary cross entropy
# evaluation metric will be accuracy
new_model.compile(optimizer=adam_opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

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

Y_pred = new_model.predict_generator(validation_generator, validation_generator.samples // BS + 1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Fake', 'Real']
print(classification_report(validation_generator.classes, y_pred))