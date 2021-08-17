


import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
#import tensorflow as tf
from PIL import Image



import sys
import keras
print('Python:{}'.format(sys.version))
print('Keras:{}'.format(keras.__version__))
# Loading of data set
(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()

print('Training Images: {}'.format(X_train.shape))
print('Testing Images: {}'.format(X_test.shape))

for i in range(0,9):
    plt.subplot(330 + 1 + i)
    img = X_train[i]
    plt.imshow(img)
plt.show()

# Normalizing input vector values from 0-255 to 0.0-1.0
np.random.seed(6)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

print(X_train[0].shape)


# One hot encoding
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_class = Y_test.shape[1]
print(num_class)

print(Y_train.shape)
print(Y_train[0])


from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAvgPool2D
from keras.optimizers import SGD

def allcnn(weights=None):
    model = Sequential()
    model.add(Conv2D(96,(3, 3), padding='same', input_shape=(3,32,32)))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3, 3),padding='same',strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192,(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3, 3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192,(3, 3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1, 1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10,(1, 1),padding='valid'))

    model.add(GlobalAvgPool2D())
    model.add(Activation('softmax'))

    if weights:
        model.load_weights(weights)
    return model

# Hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

weights = 'all_cnn_weights_0.9088_0.4994.hdf5'
model = allcnn(weights)
sgd = SGD(lr=learning_rate,decay=weight_decay,momentum= momentum,nesterov=True)
model.compile(loss='categorical_crossetnropy',optimizer=sgd,metrics=['accuracy'])
print(model.summary())

#epochs = 350
#batch_size = 32
#model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=epochs,batch_size=batch_size,verbose=1)

scores = model.evaluate(X_test,Y_test,verbose=1)
print('Accuracy: {}'.format(scores[1]))

classes = range(0,10)
names = ['airplane',
         'automobile',
         'bird',
         'cat',
         'deer',
         'dog',
         'frog',
         'horse',
         'ship',
         'truck']

class_labels = dict(zip(classes,names))
print(class_labels)

batch = X_test[100:109]
labels = np.argmax(Y_test[100,109],axis=-1)
predictions = model.predict(batch,verbose=1)

print(predictions)
print(predictions.shape)

for image in predictions:
    print(np.sum(image))

class_result = np.argmax(predictions,axis=-1)
print(class_result)

fig, axs = plt.subplot(3,3,figsize = (15,6))
fig.subplots_adjust(hspace=1)
axs = axs.flatten()

for i, img in enumerate(batch):
    for key,value in class_labels.items():
        if class_result[i] == key:
            title = 'Prediction: {} \n Actual: {}'.format(class_labels[key],class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
    axs[i].imshow(img.transpose([1,2,0]))

plt.show()



