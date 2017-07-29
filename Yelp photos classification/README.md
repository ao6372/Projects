# Yelp photo classification via Convolutional Neural Networks Transfer Learning

## Abstract

[Slides](./Capstone_Project_Slides.pdf)

<a href="https://youtu.be/WMcGYdezf7E
" target="_blank"><img src="http://img.youtube.com/vi/WMcGYdezf7E/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


This is my capstone project of Galvanize. I built a Yelp photo classification model using convolutional neural network transfer learning based on [VGG-16 model](https://arxiv.org/pdf/1409.1556.pdf). This model beat [Yelp’s own classifier](https://engineeringblog.yelp.com/2015/10/how-we-use-deep-learning-to-classify-business-photos-at-yelp.html) in precision by 1.5% (95.5% vs 94%), and in recall by 25.4% (95.4% vs 70%). A weighted-average model stacking was employed to push the final 1.5% improvement in precision. The model was built using the Keras API with the TensorFlow backend, and was trained at GPU instances at Amazon Web Services (AWS).


## Motivation 

Yelp released a new feature labeling photos updated by customers. When searching for a new restaurant, customers can easily
find information they need, e.g. menu, inside environment. Higher flow of customers are expected with this new feature.
However, [Yelp’s own classifier](https://engineeringblog.yelp.com/2015/10/how-we-use-deep-learning-to-classify-business-photos-at-yelp.html) was reported with overall precision of 94% and recall of 70%, which is still not good enought. In this 
project, I will build my model using convolutional neural network transfer learning to improve the performance of classifiers.
<img width="1092" alt="screen shot 2017-07-06 at 1 37 55 pm" src="https://user-images.githubusercontent.com/25883937/27927249-c4e062fc-6250-11e7-9dc1-157b8a80989b.png">


## Dataset
Dataset was downloaded from [Yelp Dataset Challenge Round 9](https://www.yelp.com/dataset_challenge). There are 80,000 photos with 5-category labels: menu, food, drink, inside, and outside.

The dataset was splitted to training data (70%), validation data (15%), and test data (15%).

<p align="center">
  <img width="537" alt="screen shot 2017-07-06 at 1 38 22 pm" src="https://user-images.githubusercontent.com/25883937/27927287-e2cce290-6250-11e7-85b4-b5c2ae634d52.png">
</p>



### Preprocessing

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

## Sources
I set up a p2.xlarge 
[GPU instance](https://aws.amazon.com/blogs/aws/new-p2-instance-type-for-amazon-ec2-up-to-16-gpus/)
with the GPU memory of 12 GB at Amazon Web Services (AWS) with 100 times faster computation.

<p align="center">
  <img width="589" alt="screen shot 2017-07-06 at 1 38 35 pm" src="https://user-images.githubusercontent.com/25883937/27927286-e2ca6c5e-6250-11e7-8341-65ee21d11169.png">
</p>


## Algorithm

### Introduction to VGG-16 model 

[VGG-16 model](https://arxiv.org/pdf/1409.1556.pdf) is a extremely deep network with 16 weight layers, including
13 convolutional layers and 3 full-connected layers. It was developed by [Visual Geometry Group at University of Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) for ImageNet Challenge. It performs very well with the 1000-category dataset.

<img width="896" alt="screen shot 2017-07-29 at 10 31 17 am" src="https://user-images.githubusercontent.com/25883937/28745969-2c1d31f2-7449-11e7-96df-b39dd4229a23.png">

Picture borrowed from this [blog](https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5/).

### Transfer learning

Here is [a good introduction video to transfer learning given by Prof Guestrin from University of Washington](https://youtu.be/HVbUD9aA_Ys)

The basic idea is that knowledge extracted from pre-trained model can be used in similar cases. 
For example, in this case, I borrowed parameters from the convolutional layers (as well as some 
max-pooling layers) of pre-trained VGG-16 model, and added 2 new full-connected layers, which 
will be trained with Yelp training dataset to 
get the new model. The benefit is that firstly, models can be trained very fast, about 3 hours here versus
3 weeks for the VGG-16 model. Secondly, transfer learning has proved to be powerful in many cases.

<img width="1400" alt="screen shot 2017-07-06 at 1 26 45 pm" src="https://user-images.githubusercontent.com/25883937/27926772-2f5523e0-624f-11e7-9e7a-168988a6f0b7.png">


### Train fully-connected layers

I employed [Keras API with the TensorFlow backend](https://keras.io/applications/#vgg16).

The code of new model building is here:
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

#Create my own model 
yelp_vgg16_model = Model(input=vgg16_yelp_input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
yelp_vgg16_model.summary()
```

The summary of new model will be:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
image_input (InputLayer)     (None, 128, 128, 3)       0         
_________________________________________________________________
vgg16 (Model)                multiple                  14714688  
_________________________________________________________________
flatten (Flatten)            (None, 8192)              0         
_________________________________________________________________
fc1 (Dense)                  (None, 500)               4096500   
_________________________________________________________________
predictions (Dense)          (None, 5)                 2505      
=================================================================
Total params: 18,813,693
Trainable params: 18,813,693
Non-trainable params: 0
```

### Parameter optimization

#### Epoch Number Selection
Here I select [Stochastic gradient descent (sgd) optimizer](https://keras.io/optimizers/#sgd). One epoch is defined as one forward pass and one backward pass of all the training examples.

Here I plot error curves with the number of epochs at training data and validation data, respectively. As expected, in the training curve, error keeps droping with epoch, while in the validation cruve, error becomes flat starting at epoch = 6, and rebounds at epoch = 9, indicating overfitting. So I finalize epoch = 6.

<p align="center">
  <img width="537" alt="screen shot 2017-07-06 at 1 38 22 pm" src="https://user-images.githubusercontent.com/25883937/27926567-70b1f40e-624e-11e7-8827-900ee5ad5406.png">
</p>


#### Batch Size Selection
Here error curves at vaidation data with differet batch size are compared. It is noticed that, qualitatively, smaller batch size curve converge to lower error level much faster. Since the lowest error achieved at batch size = 5 is similar to that at batch size = 10. I stop decreasing batch size and finalize batch size = 5.

<p align="center">
  <img width="537" alt="screen shot 2017-07-06 at 1 38 22 pm" src="https://user-images.githubusercontent.com/25883937/27926634-9fa4455a-624e-11e7-9124-2815f17ffeaf.png">
</p>


### Model stacking

In the following figure, I display validation accuracy of four raw models with different preprocess ways in each category and the whole dataset. Please noticed that the lower limit of y scale is from 0.85, so all models work not bad. I also noticed that every model has its own favorite, for example, model 3 (purple) performs very well in category of inside, but not satisfying enough in category of outside. Moreover, all 4 models give rise to about 94% accuracy in the whole dataset. To give the last push of the performance, I employed a model stacking based on weighted average. Actually, for each photo, every model predicts a probability for each category. I performed a grid search to get a optimal weight combination which can maximize the accuracy in the whole validation dataset.

Here is the code:
```
# There are 4 raw model: i, j, k, m
predict_combine = []
for i in xrange(0,100):
    for j in xrange(0,101-i):  
        for k in xrange(0,101-i-j): 
            i_ratio = i/100.0
            j_ratio = j/100.0
            k_ratio = k/100.0
            m_ratio = 1 - i_ratio - j_ratio - k_ratio
            temp = predict_prop_valid_v0 * i_ratio \
                    + predict_prop_valid_v1 * j_ratio \
                    + predict_prop_valid_v2 * k_ratio \
                    + predict_prop_valid_v3 * m_ratio
            y_pred = np.argmax(temp, axis=1)
            accuracy_temp = np.sum(y_pred == y_valid)/ 12500.0

            if i == 0 and j == 0 and k == 0:
                accuracy_max = accuracy_temp
                i_max = 0
                j_max = 0
                k_max = 0
            else:
                if accuracy_temp > accuracy_max:
                    accuracy_max = accuracy_temp
                    i_max = i
                    j_max = j
                    k_max = k

print "i_max = %d, j_max = %d, k_max = %d, accuracy_max = %f" \
        %(i_max, j_max, k_max, accuracy_max)
```

The stacking model increase the accuracy in the whole validation by 1.5% to 95.5%.

<img width="1176" alt="screen shot 2017-07-06 at 1 31 04 pm" src="https://user-images.githubusercontent.com/25883937/27926945-c5166830-624f-11e7-8de0-a0efc09d1226.png">


## Results

### Model's performance of 5 examples of each category

The final model can correctly predict the category for each example
<img width="1106" alt="screen shot 2017-07-29 at 5 08 47 pm" src="https://user-images.githubusercontent.com/25883937/28748567-aa4489e0-7480-11e7-8eda-94318392dc17.png">


### Confusion matrix 

The large number in the diagonal of the confusion matrix indicates very good performance of the model.
<p align="center">
  <img width="537" alt="screen shot 2017-07-06 at 1 38 22 pm" src="https://user-images.githubusercontent.com/25883937/27881123-24b88052-618d-11e7-98f0-7f569d064e9a.png">
</p>

The final metrics shows that the model here beat Yelp’s classifier in precision by 1.5% (95.5% vs 94%), and in recall by 25.4% (95.4% vs 70%).

<img width="454" alt="screen shot 2017-07-29 at 5 17 33 pm" src="https://user-images.githubusercontent.com/25883937/28748619-db1f6c00-7481-11e7-8f84-53ac68047f4c.png">



## Summary
- Train a photo classifier based on convolutional neural network transfer learning algorithm

- Improve the precision and recall compared to Yelp's classifier

- Mislabel could be one reason limiting the further improvement

- Other base models or model stacking methods may help more


## Reference
K. Simonyan, A. Zisserman
Very Deep Convolutional Networks for Large-Scale Image Recognition  
[arXiv 1409.1556, 2014](https://arxiv.org/pdf/1409.1556.pdf)

