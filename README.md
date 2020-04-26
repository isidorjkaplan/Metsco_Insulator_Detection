# Metsco_Insulator_Detection
A project for METSCO Energy Solutions Inc in to identify insulator damage from photos. Developed by first year students at the University of Toronto as part of a project in Engineering Stratagies and Practices (APS112). 

Adapted code from the following resources:
  https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d
  https://www.tensorflow.org/tutorials/images/segmentation
  https://keras.io/visualization/
  https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    
Data labelled with: https://labelbox.com/

Folder With Images: https://drive.google.com/drive/folders/1IvYfHo5RaZGpqLwC2_bD4qTeVtPvj-Uh?usp=sharing 
Within that folder there are two folders, the first has training data and the second has a sample run of our program. 

Training Data: 
  For the training data it is not seperated into training and validation in that folder, it is just the entire dataset. When we were training our code automatically takes a percentage of those images and sets them aside for the validation set while the rest is used for training. Inside the "Training Data" folder there are three other folders. The folder "[Final] Sorted Insulators" is the data provided to us by Youseff which we sorted by number of insulators and then by broken/not-broken. The folder "Manually Individually Cropped Photos" is the data that we used when training a neural network to identify damage off cropped photos (that is the dataset it got %90+ on for identifying damage). And the folder "Segmentation Network Training Data" has some images of insulators in the "Input" folder and manually labeled heatmaps in the "Target Output" folder. This is what we used to train the segmentation network. 

Sample Run: 
  I thought it would be helpful to include a folder with a sample run of our segmentation network and cropping on images that are already zoomed. The "Input" folder has a bunch of images that were inputted into the segmentation network and then cropper. In this case the input is already sorted into broken and not broken, this is just beacuse that is how the data was provided to us and has no bearing on how the actual segmentation network works. The "Segmentation Network Output" has two folders, "Broken" and "Healthy" and they contain the automatically generated heatmaps from the segmentation network on the images with the same name from the "Input" folder. After that there is the "Cropping Output" folder which is the output of the cropper using the heatmap and original images, and once again the files have the same name as their respective input files. Since the cropper breaks one image into multiple individual insulators if the input image was "image5.jpg" the output will have "image5_1.jpg, image5_2.jpg, ..." to name the seperated files. 

Source Code: 
  https://github.com/isidorjkaplan/Metsco_Insulator_Detection 
The home folder for the GitHub is listed above. Ignore the "data" folder there, it's contents are meaningless. The main things to look at there is the source code. The two main folders here are "final design" and "old designs." 

Final Design: 
  In the final design folder the file (https://github.com/isidorjkaplan/Metsco_Insulator_Detection/blob/master/final_design/network.py) is used to train the segmentation network. It creates a new network and then uses the training data to train the segmentation network to generate heatmaps and saves the weights of the network to a file in the same folder. The other important file is (https://github.com/isidorjkaplan/Metsco_Insulator_Detection/blob/master/final_design/program.py) which takes in a folder of images (In this case already sorted by broken or not broken for convienece since I needed it sorted when I was using the output to train a classifier, but this can be easily modified and does not rely on it being sorted in any way). It then takes the images that were inputted and for each image generates a heatmap, saves the image and heatmap to a file (this was used in the "Sample Run" above) and then it crops it and saves the cropped images to another file. Currently the code will just save all the images to the folder but it can be modified later to re-route images it failed to individually crop to their own folder (although it does not currently do this). And lastly, here is the code for training the classifier (https://github.com/isidorjkaplan/Metsco_Insulator_Detection/blob/master/old_designs/broken_classifier_cropped/broken_classifier_cropped.py). The code for training a classifier is the same for pretty much all of our designs, the only difference is the files that are inputted into the classifier. In the first two designs we inputted uncropped images whereas in the third design the input images are already cropped, although the actual code for the classifier is the same reguardless of what the input images look like which is why it is stored in the old designs folder.
