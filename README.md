# Salary-Prediction
To predict whether the salary of an employee exceeds 50K or will be less than 50K.
# Introduction
According to the dataset under consideration, we need to predict the categorical feature ‘salary’ that is categorized into two, a salary <=50k and a salary >50k. The dataset is a mixture of categorical and continuous features where other categorical features include, 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'. Also, this dataset contains missing values as well as outliers which are then sorted out. Categorical features are label encoded and analysed. Various exploratory data analysis techniques and visualizations were done on the dataset.
# Dataset
The dataset used here is the Adult Income dataset consisting of 32561 observations and 14 attributes where the aim is to predict whether an employee salary exceeds 50K or not.
# Approach
•	Import the HR dataset.<br/> 
•	Start an exploratory data analysis to detect the key factors, trends and patterns in the dataset.<br/>
•	Apply various data cleaning techniques to prepare the dataset. The data cleaning techniques include finding missing values, outliers and duplicates if any.<br/>
•	Try various visualization techniques with the data in hands. Elaborate the dataset for the training and testing phase and try classification model, since the predictor variable ‘salary’ is a categorical feature.<br/>
•	After getting the best model, design a website for salary prediction using flask.<br/>
# Coclusion
After building various machine learning models, Gradient Boosting Classifier stood first in terms of accuracy in prediction with 86.90% when compared with other models. Hypertuning was also done on the best model selected. 
![image](https://user-images.githubusercontent.com/79460483/111096425-838a4480-8565-11eb-9fa7-5779d4b925f0.png)
