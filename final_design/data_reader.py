import json
from urllib import request
from PIL import Image
import numpy as np
import scipy.misc
import os


def normalize_image(file):
    img = Image.open(file, 'r')
    arr = np.array(img)
    arr = np.floor_divide(arr, 255)
    img = Image.fromarray(arr)
    img.save(file)


with open('data.json') as f:
  data = json.load(f)


# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
folder = '/home/isidor/Documents/keras/data/mask_data/'
os.system("find " + folder + " -name '*.jpg' -delete")
os.system("find " + folder + " -name '*.png' -delete")
print("Deleted old files")
masks = ['a','b','c', 'd', 'e', 'f']
i = 0
while i < len(data):
    if len(data[i]['Label']) != 0:
        image = data[i]['Labeled Data']
        request.urlretrieve(image, folder + 'frames/files/image' + str(i) + '.jpg')

        arr = 0
        numOfObjects = len(data[i]['Label']['objects'])
        for j in range(0,numOfObjects):
            mask = data[i]['Label']['objects'][j]['instanceURI']
            mask_file = folder + 'tmp_masks/image' + str(i) + masks[j] + '.png'
            request.urlretrieve(mask, mask_file)
            img = Image.open(mask_file)
            if j == 0:
                arr = np.asarray(img)
            else:
                arr = arr + np.asarray(img)
        img = Image.fromarray(arr)
        img.save(folder + 'masks/files/image' + str(i) + '.png')
        #normalize_image(mask_file)
    i = i+1