# Abstract

Here is [the first version of slides](./Capstone_Project_slides.pdf).

[Presentation vedio in YouTube](https://youtu.be/WMcGYdezf7E)

This is my capstone project of Galvanize. I built a Yelp photo classification model using convolutional neural network transfer learning based on [VGG-16 model](https://arxiv.org/pdf/1409.1556.pdf). This model beat [Yelp’s own classifier](https://engineeringblog.yelp.com/2015/10/how-we-use-deep-learning-to-classify-business-photos-at-yelp.html) in precision by 1.5% (95.5% vs 94%), and in recall by 25.4% (95.4% vs 70%)


– Beat Yelp’s classifier in precision by 1.5% (95.5% vs 94%), and in recall by 25.4% (95.4% vs 70%)
– Built deep learning models using the Keras API with the TensorFlow backend
– Set up GPU instances at Amazon Web Services (AWS) with 100 times faster computation



Built a Yelp photo classification model using convolutional neural network transfer learning based on VGG-16 model.

There are categories: menu, food, drink, inside, and outside.

This model beat Yelp's own classifier in precision by 1.5% (95.5% vs 94%), and in recall by 25.4% (95.4% vs 70%). 

# Motivation 



# Data set


## Preprocessing


# Algorithm

## Introduction to VGG-16 model


# Result

## Confusion matrix 

![confusion matrix](https://user-images.githubusercontent.com/25883937/27881123-24b88052-618d-11e7-98f0-7f569d064e9a.png)

# Summary
– Train a photo classifier based on CNN transfer learning algorithm
– Improve the precision and recall 
– Mislabel could be one reason limiting the further improvement
-- Other base models or model stacking methods may help more


# Reference
K. Simonyan, A. Zisserman
Very Deep Convolutional Networks for Large-Scale Image Recognition  
[arXiv 1409.1556, 2014](https://arxiv.org/pdf/1409.1556.pdf)

