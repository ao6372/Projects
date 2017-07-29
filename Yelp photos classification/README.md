# Abstract

[Slides](./Capstone_Project_Slides.pdf)

<a href="https://youtu.be/WMcGYdezf7E
" target="_blank"><img src="http://img.youtube.com/vi/WMcGYdezf7E/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


This is my capstone project of Galvanize. I built a Yelp photo classification model using convolutional neural network transfer learning based on [VGG-16 model](https://arxiv.org/pdf/1409.1556.pdf). This model beat [Yelp’s own classifier](https://engineeringblog.yelp.com/2015/10/how-we-use-deep-learning-to-classify-business-photos-at-yelp.html) in precision by 1.5% (95.5% vs 94%), and in recall by 25.4% (95.4% vs 70%). A weighted-average model stacking was employed to push the final 1.5% improvement in precision. The model was built using the Keras API with the TensorFlow backend, and was trained at GPU instances at Amazon Web Services (AWS).


# Motivation 



<img width="1092" alt="screen shot 2017-07-06 at 1 37 55 pm" src="https://user-images.githubusercontent.com/25883937/27927249-c4e062fc-6250-11e7-9dc1-157b8a80989b.png">


# Dataset
There are categories: menu, food, drink, inside, and outside.


<p align="center">
  <img width="422" alt="screen shot 2017-07-06 at 1 38 12 pm" src="https://user-images.githubusercontent.com/25883937/27927285-e2c9f80a-6250-11e7-9553-e8fdd427730e.png"> 
</p>

test1

<p align="center">
  <img width="537" alt="screen shot 2017-07-06 at 1 38 22 pm" src="https://user-images.githubusercontent.com/25883937/27927287-e2cce290-6250-11e7-85b4-b5c2ae634d52.png">
</p>

test2

<p align="center">
  <img width="589" alt="screen shot 2017-07-06 at 1 38 35 pm" src="https://user-images.githubusercontent.com/25883937/27927286-e2ca6c5e-6250-11e7-8341-65ee21d11169.png">
</p>

## Preprocessing


# Algorithm

## Introduction to VGG-16 model 

<img width="896" alt="screen shot 2017-07-29 at 10 31 17 am" src="https://user-images.githubusercontent.com/25883937/28745969-2c1d31f2-7449-11e7-96df-b39dd4229a23.png">

Picture borrowed from this [blog](https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5/).

## Transfer learning

[A good video of introduction to transfer learning given by Prof Guestrin from University of Washington](https://youtu.be/HVbUD9aA_Ys)

<img width="1400" alt="screen shot 2017-07-06 at 1 26 45 pm" src="https://user-images.githubusercontent.com/25883937/27926772-2f5523e0-624f-11e7-9e7a-168988a6f0b7.png">


## Train fully-connected layers


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

