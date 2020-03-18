from keras.preprocessing.image import ImageDataGenerator
from keras_segmentation.models.unet import vgg_unet
from keras import backend as K
import os, os.path


img_width = img_height = 32*2*2
data_folder = '/home/isidor/Documents/keras/data/mask_data'
batch_size = 1
epoches = 8
nb_train_samples = len(os.listdir(data_folder + "/frames/"))

model = vgg_unet(n_classes=2 ,  input_height=img_width, input_width=img_height  )
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy', 'mse'])


model.train(train_images = data_folder + "/frames/",
            train_annotations= data_folder + "/masks/",
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epoches)
