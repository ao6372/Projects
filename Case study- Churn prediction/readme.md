# Summary
A one-day case study results of Churn prediction based on a ride-sharing company data are presented here, including a [jupyter notebook](./case_study.ipynb) and a powerpoint(./case_study_slides.key).

## Code 
In the [jupyter notebook](./case_study.ipynb), following steps are included:

    -- Firstly I performed exploratory data analysis (EDA)
  
    -- In the data engineering section, null values were filled 
  
    -- Categorical features were converted to numeric ones or replaced by several dummy columns
  
    -- A new column ['Churn?'] was created as the lable based on the date column, and two corresponding 
       date columns were removed to avoid the data leakage
  
    -- Studied Feature Importances using Random Forest and make the most important feature "avg_dist" as 
       an example to analyze from a business perspective (see [powerpoint] for detail)
