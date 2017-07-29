# Abstract

[Slides](./Capstone_Project_Slides.pdf)

<a href="https://youtu.be/WMcGYdezf7E
" target="_blank"><img src="http://img.youtube.com/vi/WMcGYdezf7E/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


This is my capstone project of Galvanize. I built a Yelp photo classification model using convolutional neural network transfer learning based on [VGG-16 model](https://arxiv.org/pdf/1409.1556.pdf). This model beat [Yelp’s own classifier](https://engineeringblog.yelp.com/2015/10/how-we-use-deep-learning-to-classify-business-photos-at-yelp.html) in precision by 1.5% (95.5% vs 94%), and in recall by 25.4% (95.4% vs 70%). A weighted-average model stacking was employed to push the final 1.5% improvement in precision. The model was built using the Keras API with the TensorFlow backend, and was trained at GPU instances at Amazon Web Services (AWS).


# Motivation 

Yelp released a new feature labeling photos updated by customers. When searching for a new restaurant, customers can easily
find information they need, e.g. menu, inside environment. Higher flow of customers are expected with this new feature.
However, [Yelp’s own classifier](https://engineeringblog.yelp.com/2015/10/how-we-use-deep-learning-to-classify-business-photos-at-yelp.html) was reported with overall precision of 94% and recall of 70%, which is still not good enought. In this 
project, I will build my model using convolutional neural network transfer learning to improve the performance of classifiers.
<img width="1092" alt="screen shot 2017-07-06 at 1 37 55 pm" src="https://user-images.githubusercontent.com/25883937/27927249-c4e062fc-6250-11e7-9dc1-157b8a80989b.png">


# Dataset
Dataset was downloaded from [Yelp Dataset Challenge Round 9](https://www.yelp.com/dataset_challenge). There are 80,000 photos with 5-category labels: menu, food, drink, inside, and outside.

The dataset was splitted to training data (70%), validation data (15%), and test data (15%).

<p align="center">
  <img width="537" alt="screen shot 2017-07-06 at 1 38 22 pm" src="https://user-images.githubusercontent.com/25883937/27927287-e2cce290-6250-11e7-85b4-b5c2ae634d52.png">
</p>



## Preprocessing

Firstly, I reshape each photo to 128x128x3(RGB), and if needed, rotate or vertical/horizontal flip photos to get more 
training data with different orientations. 
```
def preprocess_input(input_file, outfile_name, dimension_width = 128):
    ##input shape 500 x 49152
    num_row = input_file.shape[0]
    train_otsd_test1 = np.reshape(input_file, (num_row, dimension_width, dimension_width, 3))
    datagen = ImageDataGenerator(vertical_flip=True, 
                               horizontal_flip=True,
                               rotation_range=90)
    datagen.fit(train_otsd_test1)
    # configure batch size and retrieve one batch of images
    for X_batch in datagen.flow(train_otsd_test1, batch_size=num_row, shuffle=False):    
        X_batch2 = np.reshape(X_batch, (num_row, -1))
        np.save(outfile_name, X_batch2)
        break
```
<p align="center">
  <img width="422" alt="screen shot 2017-07-06 at 1 38 12 pm" src="https://user-images.githubusercontent.com/25883937/27927285-e2c9f80a-6250-11e7-9553-e8fdd427730e.png"> 
</p>

# Sources
I set up a p2.xlarge 
[GPU instance](https://aws.amazon.com/blogs/aws/new-p2-instance-type-for-amazon-ec2-up-to-16-gpus/)
with the GPU memory of 12 GB at Amazon Web Services (AWS) with 100 times faster computation.

<p align="center">
  <img width="589" alt="screen shot 2017-07-06 at 1 38 35 pm" src="https://user-images.githubusercontent.com/25883937/27927286-e2ca6c5e-6250-11e7-8341-65ee21d11169.png">
</p>


# Algorithm

## Introduction to VGG-16 model 

[VGG-16 model](https://arxiv.org/pdf/1409.1556.pdf) is a extremely deep network with 16 weight layers, including
13 convolutional layers and 3 full-connected layers. It was developed by [Visual Geometry Group at University of Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) for ImageNet Challenge. It performs very well with the 1000-category dataset.

<img width="896" alt="screen shot 2017-07-29 at 10 31 17 am" src="https://user-images.githubusercontent.com/25883937/28745969-2c1d31f2-7449-11e7-96df-b39dd4229a23.png">

Picture borrowed from this [blog](https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5/).

## Transfer learning

Here is [a good introduction video to transfer learning given by Prof Guestrin from University of Washington](https://youtu.be/HVbUD9aA_Ys)

The basic idea is that knowledge extracted from pre-trained model can be used in similar cases. 
For example, in this case, I borrowed parameters from the convolutional layers (as well as some 
max-pooling layers) of pre-trained VGG-16 model, and added 2 new full-connected layers, which 
will be trained with Yelp training dataset to 
get the new model. The benefit is that firstly, models can be trained very fast, about 3 hours here versus
3 weeks for the VGG-16 model. Secondly, transfer learning has proved to be powerful in many cases.

<img width="1400" alt="screen shot 2017-07-06 at 1 26 45 pm" src="https://user-images.githubusercontent.com/25883937/27926772-2f5523e0-624f-11e7-9e7a-168988a6f0b7.png">


## Train fully-connected layers

I employed [Keras API with the TensorFlow backend](https://keras.io/applications/#vgg16).

The code is here:
```
#Get back the convolutional part of a VGG network trained on ImageNet
#"include_top = False" exclude the 3 fully-connected layers at the top of the network.
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

#Print out the model summary
model_vgg16_conv.summary()

#Create my own input format (here 128x128x3)
vgg16_yelp_input = Input(shape=(128, 128, 3),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(vgg16_yelp_input)

#Add two fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(500, activation='relu', name='fc1')(x)

#There 5 neurons in the last output layer due to 5-category output
x = Dense(5, activation='softmax', name='predictions')(x)
```


![error_curve_big2](https://user-images.githubusercontent.com/25883937/27926567-70b1f40e-624e-11e7-8827-900ee5ad5406.png)


![validation](https://user-images.githubusercontent.com/25883937/27926634-9fa4455a-624e-11e7-9124-2815f17ffeaf.png)

## Model stacking

<img width="1176" alt="screen shot 2017-07-06 at 1 31 04 pm" src="https://user-images.githubusercontent.com/25883937/27926945-c5166830-624f-11e7-8de0-a0efc09d1226.png">


# Result

## Confusion matrix 

![confusion matrix](https://user-images.githubusercontent.com/25883937/27881123-24b88052-618d-11e7-98f0-7f569d064e9a.png)

# Summary
- Train a photo classifier based on CNN transfer learning algorithm

- Improve the precision and recall 

- Mislabel could be one reason limiting the further improvement

- Other base models or model stacking methods may help more


# Reference
K. Simonyan, A. Zisserman
Very Deep Convolutional Networks for Large-Scale Image Recognition  
[arXiv 1409.1556, 2014](https://arxiv.org/pdf/1409.1556.pdf)

