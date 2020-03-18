from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

batch_size = 1
img_width = 32
img_height = img_width

val_datagen = ImageDataGenerator(rescale=1. / 255)
data_folder = '/home/isidor/Documents/keras/data/mask_data'

train_image_generator = train_datagen.flow_from_directory(

    data_folder + '/frames',
    batch_size=batch_size, target_size=(img_width, img_height),class_mode=None)

train_mask_generator = train_datagen.flow_from_directory(
    data_folder + '/masks',
    batch_size=batch_size, target_size=(img_width, img_height),class_mode=None)

val_image_generator = val_datagen.flow_from_directory(
    data_folder + '/frames',
    batch_size=batch_size,target_size=(img_width, img_height),class_mode=None)

val_mask_generator = val_datagen.flow_from_directory(
    data_folder + '/masks',
    batch_size=batch_size, target_size=(img_width, img_height),class_mode=None)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)










#Training
from keras.models import Sequential

nb_train_samples = 1
nb_validation_samples = 1
epoches = 2


from keras import backend as K
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=2 ,  input_height=img_height, input_width=img_width )

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy', 'mse'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epoches,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)
