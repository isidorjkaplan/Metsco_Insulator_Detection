from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
import numpy as np
from PIL import Image

def printHistory(history):
    # Plot history: MSE
    plt.plot(history.history['iou_score'], label='Score (testing data)')
    #plt.plot(history.history['val_iou_score'], label='Score (validation data)')
    plt.title('Accuracy on segmenting insulators')
    plt.ylabel('IOU Score')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


img_width = img_height = 32*4
epoches = 300
data_folder = '/home/isidor/Documents/keras/data/mask_data'
steps_per_epoch = 5
batch_size = 3#int(len(os.listdir(data_folder + "/frames/files/"))/steps_per_epoch)
net_file = "segment.h5"

def show(train_generator, model):
    N = 5
    f,ax = plt.subplots(N, 3)
    for i in range(0, N):
        (img, mask) = train_generator.__next__()
        out = model.predict(img)
        out = out[0].reshape((img_width, img_height))

        ax[i,0].imshow(img[0])
        ax[i,1].imshow(out)
        ax[i,2].imshow(mask[0])
    plt.show()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_image_generator = train_datagen.flow_from_directory(
    data_folder + '/frames',
    batch_size=batch_size, target_size=(img_width, img_height),class_mode=None, seed=1, subset='training')

train_mask_generator = train_datagen.flow_from_directory(
    data_folder + '/masks',
    batch_size=batch_size, target_size=(img_width, img_height),class_mode=None, seed=1, subset='training')

train_generator = zip(train_image_generator,train_mask_generator)

BACKBONE = 'resnet34'

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')
opt = Adam(lr=0.001)
model.compile(opt, loss=bce_jaccard_loss, metrics=[iou_score])

model.load_weights(net_file)
model.summary()


#show(train_generator,model)
#show_validation(test_image_generator, model)


history = model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch, epochs=epoches)
model.save_weights(net_file)
printHistory(history)

show(train_generator,model)







