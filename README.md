# Teeth-diseases-Classification
This repo explains how to implement a CNN to classify 7 Teeth Diseases.
A ResNet50 is biult from scratch and fine tuned to calssify 7 dental diseaeses.
first and like any NN model I started with data pre-processing to prepare the dental images for analysis through normalization, augmentation. This will ensure the images are in optimal condition for model training and evaluation.
 Visualize the distribution of the classes to understand the balance of the dataset.
 Display images before and after augmentation to evaluate the effectiveness of preprocessing techniques and ensure the transformations are enhancing the dataset appropriately
I found that the data was not totally balanced so I assigned weights for each class.
The next step is to biuld the model which I used 3 functions to implement it :
    1- the identity functuion: preserves the spatial dimensions while mapping the input with output 
    2- the convolutional block : changes the spatial dimensions while also having a shortvut connection
        which undergoes its own convolutional transformations to match the dimensions of its path 
    3- the ResNet50 function which includes a number of transformations like identity blocks , convolutional blocks...etc and       finaly fine tuning it with an output layer with 7 units and 'softmax' activiaion function 
  I also implemented some techiques like early stopping and rduce lr plateau to help me developing the model's performance
trained the model with 250 eochs resulting about 99% validation accuracy and 99.5% evaluating accuracy 
