# Maintenance cost reduction through a failure-prediction Neural Network Model

## Background
A 3D Technologies company has a series of sensors which deliver daily monitor data. Predictive models are necessary to help predict the working conditions of devices to indicate when to perform maintenance. This techniques will save cost by avoiding failure of devices or not replacing healthy devices far early from true failure. 

## Data Exploration
- Highly imbalanced dataset (0.09% failure data)
- About 10% devices has failed report
- For each device, there is one failure at most (the latest date), or there is no failure
- Based on scatter matrix, attribute 7 and 8 are highly linear-related

## Data Engineering
### - Highly imbalanced dataset (0.09% failure data) is always a big challenging problem in machine learning. Oversampling and under sampling are two popular methonds to handle imbalanced cases. Although random under sampling works in many cases, it is not a good option in time series data, e.g. in this project. Here, I will used TARGETED under sampling, i.e. for each device, the 6th, 7th, and last date data are kept, because 
- (a) in the next step new features creation is based on 5 days info before each specific date;
- (b) features on early dates far away from the last potential failure date is more typical health data than those very close to the last date, which have similar performance to the last-day date, based on above plots of feature performance exploration;
- (c) since failure data only happened on the last day, all failure data are kept, while the number of health data drops significantly. 
After under sampling, the percentage of failure data increases by 2 orders to 4%.
### - Attribute 7 are removed due to its exact linear relationship with attribute 8
### - Devices with total number of date fewer than 10 are removed
### - For specific date of each device, normalize each feature to [0, 1], only taking into account all data before and including that specific date data
### - Create two new types of features for each raw feature:
   - (a) moving average 
   - (b) standard deviation
   of data within 5 days before specific date
   
```
def normalized_new_feature_create_under_sampling(df, column_list, row_index_all = 1):
    # Perform under sampling, normalization and new feature creation  
    # row_index_all = 1 used in testing data where all rows are preprocessed
    # row_index_all = 0 used in training data where only the 6th, 7th and last row 
    # are preprocessed
    device_list = list(set(df['device']))
    print len(device_list)
    for index, device in enumerate(device_list):   
        print index
        df_temp = df[df['device'] == device]
        df_temp_raw_index = list(df_temp.index)
        df_temp = df_temp.reset_index(drop=True)
        if row_index_all == 1:
            row_index = xrange(5,len(df_temp))
        else:
            row_index = [5, 6] + [len(df_temp)-1]
                
        for i in row_index:
            for feature in column_list:
                column_value = df_temp.iloc[:i+1][feature].values               
                attribute_normal_name_temp = feature + '_normal' 
                attribute_mean_name_temp = feature + '_mean' 
                attribute_std_name_temp = feature + '_std' 
                if max(column_value) == 0:
                    df.loc[df_temp_raw_index[i], attribute_normal_name_temp] = 0
                    df.loc[df_temp_raw_index[i], attribute_mean_name_temp] = 0
                    df.loc[df_temp_raw_index[i], attribute_std_name_temp] = 0
                elif (max(column_value) != 0) and (max(column_value) == min(column_value)):
                    df.loc[df_temp_raw_index[i], attribute_normal_name_temp] = 1
                    df.loc[df_temp_raw_index[i], attribute_mean_name_temp] = 0
                    df.loc[df_temp_raw_index[i], attribute_std_name_temp] = 0                    
                else:
                    column_value_last5 = (column_value[-6:-1]-min(column_value))/\
                                    float(max(column_value)-min(column_value))
                    df.loc[df_temp_raw_index[i], attribute_normal_name_temp] = \
                            (df.loc[df_temp_raw_index[i], feature]-min(column_value))/\
                            float(max(column_value)-min(column_value))
                    df.loc[df_temp_raw_index[i], attribute_mean_name_temp] = \
                                                            np.mean(column_value_last5)
                    df.loc[df_temp_raw_index[i], attribute_std_name_temp] = \
                                                            np.std(column_value_last5)
    df.drop(column_list, axis=1, inplace=True)
    return df
```

### - Due to the very small number of failure data, the data are splitted to training data (80% devices) and testing data (20% devices), and optimization of the model parameters is based on cross-validation.

## Modeling
