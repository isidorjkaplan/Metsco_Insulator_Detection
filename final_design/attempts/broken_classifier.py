from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os


def printHistory(accuracy, val_accuracy):
    #plt.plot(history.history['mean_squared_error'], label='MSE (testing data)')
    plt.plot(accuracy, label='Accuracy (training data)')
    plt.plot(val_accuracy, label='Accuracy (validation data)')
    plt.title('Identifying damage from auto cropped images and auto-masks')
    plt.ylabel('Accuracy')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


# dimensions of our frames.
img_width = img_height = 32*4
train_data_dir = '/data/auto_generated_masks/train'
validation_data_dir = '/data/auto_generated_masks/validation'
epochs = 200
network_file = "../classifier.h5"
batch_size = 50#len(os.listdir(train_data_dir + "/frames/Broken")) + len(os.listdir(train_data_dir + "/frames/Healthy"))
val_batch_size = 50#len(os.listdir(validation_data_dir + "/frames/Broken")) + len(os.listdir(validation_data_dir + "/frames/Healthy"))
batches = 5



if K.image_data_format() == 'channels_first':
    input_shape = (4, img_width, img_height)
else:
    input_shape = (img_width, img_height, 4)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_image_generator = train_datagen.flow_from_directory(
    train_data_dir + "/frames",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary', seed = 1)

train_mask_generator = train_datagen.flow_from_directory(
    train_data_dir + "/masks",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary', seed = 1)

val_image_generator = train_datagen.flow_from_directory(
    validation_data_dir + "/frames",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary', seed = 1)

val_mask_generator = train_datagen.flow_from_directory(
    validation_data_dir + "/masks",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary', seed = 1)


def augment_batch(img_generator, mask_generator, size):
    img, label = img_generator.next()
    mask, label2 = mask_generator.next()
    size = min(size, len(label))
    input = np.zeros((size, img_height, img_width, 4))
    if max(abs(label - label2)) != 0:
        print("Input mistmatch!")
        exit(0)

    for j in range(0, size):
        input[j] = (np.concatenate((img[j], mask[j][:,:,0].reshape(img_width, img_height, 1)), axis=2))
        #print(np.max(input[i][:,:,3]))

    return input, label

def augment_batches(img_gen, mask_gen, batch_size, batches):
    input_arr = np.zeros((0, img_height, img_width, 4))
    label_arr = np.zeros(0)
    for i in range(0, batches):
        augIn, augOut = augment_batch(img_gen, mask_gen, batch_size)
        input_arr = np.concatenate((input_arr, augIn))
        label_arr = np.concatenate((label_arr, augOut))
    return input_arr, label_arr


accuracy = np.zeros(0)
val_accuracy = np.zeros(0)
for i in range(0, epochs):
    print("\nStarting real epoch " + str(i) + "/" + str(epochs))
    merged, label = augment_batches(train_image_generator, train_mask_generator, batch_size, batches)
    val_merged, val_label = augment_batches(val_image_generator, val_mask_generator, val_batch_size, batches)
    history = model.fit(
        x=merged,
        y=label,
        epochs=5,
        validation_data=(val_merged, val_label))
    accuracy = np.concatenate((accuracy, history.history['accuracy']))
    val_accuracy = np.concatenate((val_accuracy, history.history['val_accuracy']))

printHistory(accuracy, val_accuracy)

model.save_weights(network_file)
