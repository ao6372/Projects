# Maintenance cost reduction through a failure-prediction Neural Network Model

## Background
A 3D Technologies company has a series of sensors which deliver daily monitor data. Predictive models are necessary to help predict the working conditions of devices to indicate when to perform maintenance. This techniques will save cost by avoiding failure of devices or not replacing healthy devices far early from true failure. 

## Data Exploration
- Highly imbalanced dataset (0.09% failure data)
- About 10% devices has failed report
- For each device, there is one failure at most (the latest date), or there is no failure
- Based on scatter matrix, attribute 7 and 8 are highly linear-related

## Data Engineering
### Highly imbalanced dataset (0.09% failure data) is always a big challenging problem in machine learning. Oversampling and under sampling are two popular methonds to handle imbalanced cases. Although random under sampling works in many cases, it is not a good option in time series data, e.g. in this project. Here, I will used TARGETED under sampling, i.e. for each device, the 6th, 7th, and last date data are kept, because 
- (a) in the next step new features creation is based on 5 days info before each specific date;
- (b) features on early dates far away from the last potential failure date is more typical health data than those very close to the last date, which have similar performance to the last-day date, based on above plots of feature performance exploration;
- (c) since failure data only happened on the last day, all failure data are kept, while the number of health data drops significantly. 
After under sampling, the percentage of failure data increases by 2 orders to 9%.
### Attribute 7 are removed due to its exact linear relationship with attribute 8
### Devices with total number of date fewer than 10 are removed
