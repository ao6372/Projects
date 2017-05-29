# Summary
A one-day case study results of Churn prediction based on a ride-sharing company data are presented here, including a [jupyter notebook](./case_study.ipynb) and [slides](./case_study_slides.pdf).

## Code 
In the [jupyter notebook](./case_study.ipynb), following steps are included:

    -- Performed exploratory data analysis (EDA)
  
    -- Filled null values  
  
    -- Converted categorical features to numeric ones or replaced them with several dummy columns
  
    -- A new column ['Churn?'] was created as the lable based on the date column, and two corresponding 
       date columns were removed to avoid the data leakage
  
    -- Studied Feature Importances using Random Forest and make the most important feature "avg_dist" as 
       an example to analyze from a business perspective (see [powerpoint] for detail)
       
    -- Find the optimal parameters by cross-validated grid search
    
    -- Fit optimal model 
    
    -- Test model with confusion matrix and ROC curve
    
## Slides
In the [slides](./case_study_slides.pdf), I recommend solutions to improve the retention rate. Several assumptions are proposed to study a benefit matrix. At last, future plans are proposed.
    
