from keras.optimizers import Adam
from segmentation_models import Unet
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import os
from random import random

net_file = "segment.h5"
BACKBONE = 'resnet34'

source_data_folder = "/home/isidor/Documents/keras/data/old/all_insulators_broken_not_broken"
# source_validation_folder = "/home/isidor/Documents/keras/data/fully_sorted_by_broken/validation"
img_width = img_height = 32 * 4
crop_scale = 5
batch_size = len(os.listdir(source_data_folder + "/Broken")) + len(os.listdir(source_data_folder + "/Healthy"))
batches = 20
save_folder = "/home/isidor/Documents/keras/data/auto_generated_masks"
auto_cropped_folder = "/home/isidor/Documents/keras/data/auto_cropped"
os.system("find " + save_folder + " -name '*.jpg' -delete")
os.system("find " + auto_cropped_folder + " -name '*.jpg' -delete")
# This is the average pixel value for the picture
no_insulator_cutoff = 0.01

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')
opt = Adam(lr=0.001)
# model.compile(opt, loss=bce_jaccard_loss, metrics=[iou_score])
model.load_weights(net_file)

data_gen = ImageDataGenerator(rescale=1. / 255)
image_generator = data_gen.flow_from_directory(
    source_data_folder,
    batch_size=batch_size//batches, target_size=(img_width, img_height), class_mode='binary', seed=1)
crop_generator = data_gen.flow_from_directory(
    source_data_folder,
    batch_size=batch_size//batches, target_size=(img_width * crop_scale, img_height * crop_scale), class_mode='binary', seed=1)

cutoff = 0.05


def get_crop_bounds(mask_arr):
    top = bottom = left = right = 0
    for row in range(0, img_height):
        if np.max(mask_arr[row]) > cutoff:
            bottom = row
    for row in range(img_height - 1, 0, -1):
        if np.max(mask_arr[row]) > cutoff:
            top = row
    for col in range(0, img_width):
        if np.max(mask_arr[:, col]) > cutoff:
            right = col
    for col in range(img_width - 1, 0, -1):
        if np.max(mask_arr[:, col]) > cutoff:
            left = col

    return (left, top, right, bottom)


def crop_individually(mask_arr):
    insulator = False
    list = []

    top = 0
    for row in range(0, img_height):
        if not insulator:
            if np.max(mask_arr[row]) > cutoff:
                top = row
                insulator = True
        if insulator:
            if np.max(mask_arr[row]) <= cutoff:
                list.append((top, row))
                insulator = False
    return list


def rescale(box, factor):
    return (box[0] * factor, box[1] * factor, box[2] * factor, box[3] * factor)


display = []
skipped = 0
failed_to_individually_crop = 0
for batch in range(0, batches):
    print("Starting batch " + str(batch))
    img, label = image_generator.next()
    crop_img, crop_label = crop_generator.next()
    out = model.predict(img)
    for i in range(0, len(label)):
        if label[i] == 1:
            tag = "Healthy"
        else:
            tag = "Broken"

        if random() > 0.2:
            train_or_validate = "train"
        else:
            train_or_validate = "validation"

        save_file_mask = save_folder + "/" + train_or_validate + "/masks/" + tag + "/" + "image" + str(i) + ".jpg"
        save_file_img = save_folder + "/" + train_or_validate + "/frames/" + tag + "/" + "image" + str(i) + ".jpg"
        mask_image_arr = out[i].reshape((img_width, img_height))
        avg = np.average(mask_image_arr)
        if avg > no_insulator_cutoff:
            if i < 5:
                display.append((img[i], mask_image_arr))
            box = get_crop_bounds(mask_image_arr)
            mask_image = Image.fromarray((mask_image_arr * 255).astype(np.uint8)).crop(box)
            mask_image.save(save_file_mask)

            frame_image = Image.fromarray(((crop_img[i]) * 255).astype(np.uint8)).crop(rescale(box, crop_scale))
            frame_image.save(save_file_img)

            # frame_image.show()
            # exit(0)

            insulator_heights = crop_individually(mask_image_arr)
            if len(insulator_heights) <= 1:
                failed_to_individually_crop = failed_to_individually_crop + 1
            else:
                for insulator in range(0, len(insulator_heights)):
                    left = box[0]
                    right = box[2]
                    top = insulator_heights[insulator][0]
                    bottom = insulator_heights[insulator][1]
                    cropped_insulator = Image.fromarray(((crop_img[i]) * 255).astype(np.uint8)).crop(
                        rescale((left, top, right, bottom), crop_scale))

                    if label[i] == 1:
                        cropped_file = auto_cropped_folder + "/Healthy/image" + str(i) + "_" + str(insulator) + ".jpg"
                    else:
                        cropped_file = auto_cropped_folder + "/Unsorted/image" + str(i) + "_" + str(insulator) + ".jpg"
                    cropped_insulator.save(cropped_file)

        else:
            skipped = skipped + 1

print("Skipped " + str(skipped) + "/" + str(batch_size) + " due to not detecting insulator")
print("Failed to individually crop " + str(failed_to_individually_crop) + "/" + str(
    batch_size - skipped) + " due to insulator overlap")

N = 5
f, ax = plt.subplots(N, 2)
for i in range(0, N):
    picture, mask_image = display[i]
    ax[i, 0].imshow(picture)
    ax[i, 1].imshow(mask_image)
plt.show()
print("Printed 5 sample images with auto-gen masks")
