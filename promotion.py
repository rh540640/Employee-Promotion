# -*- coding: utf-8 -*-
"""Promotion.ipynb

Submitted By: 

> Richard Honey

> Keerti

<h1><center>Employee Promotion Prediction</center></h1>

`Group Project`

<h3><center>Predicting whether an employee is promoted or not based on various factors</center></h3>

The aim is to analyze the various factors that can contribute to the promotion of an employee. Based on the analysis, predict which employees will be promoted.<br>

The following details for an employee is given in the dataset :-
    
   * Department - department of the employee
   * Region - region as designated by the company
   * Education - qualification of the employee
   * Gender - gender of the employee
   * Recruitment channel - means via which employee was recruited
   * No of trainings - total number of trainings undergone by the employee
   * Age - age of the employee
   * Previous year ratings - previous year performance ratings of the employee
   * Length of service - total years worked for the company
   * KPIs met(1 if >80%) - total KPIs met in the tenure
   * Average training score - average score on trainings
   * Awards won - Awards won if any
   
The target column is the *is_promoted* column. The column is binary and specifies whether the employee was promoted or not.
"""

# Libraries to import
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

df = pd.read_csv('Promotion.csv')

"""## Data Structure"""

df.info()

df.shape

"""## Data Pre-Processing"""

cols = ['employee_id', 'department', 'region', 'education', 'gender',
       'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating',
       'length_of_service', 'KPIs_met', 'awards_won',
       'avg_training_score', 'is_promoted']
df.columns = cols

df.isnull().sum()

"""A new employee joining in the current year would not have a *previous_year_rating*. Thus, *previous_year_rating* contains 4224 null values. *Previous_year_rating* which are Null are compared to the *length_of_service* column gives the number of years employee has now worked for the company. The *length of service* has the minimum value of 1, this implies that any new joinee is also by default said to have worked for the company for one year.

### Treating null values

#### Treating null values in *previous_year_rating*
"""

df.previous_year_rating.value_counts(dropna=False)

service_filter = df[df.length_of_service == 1]
print(
    "Null rating counts of employees with length of service 1\n",
    service_filter.previous_year_rating.isnull().sum()
)
print(
    "Null rating counts of employees with length of service 1 and promoted\n",
    service_filter[service_filter.is_promoted == 1].previous_year_rating.isnull().sum()
)
df.previous_year_rating = df.previous_year_rating.fillna(0)

"""#### Treating null values in *education*"""

df[df.education.isna()].is_promoted.value_counts()

"""One solution to impute the education column is to fill the Null values with the mode of the column. The mode of the column gives the qualification that is most frequent among the employees."""

depts = df.department.unique()

for dept in depts:
    edu = df[df.department == dept].education.mode()[0]
    print(dept," : ",edu)

df.education = df.education.fillna(df.education.mode()[0])

"""#### Checking if all the null vales are cleaned"""

df.isnull().sum()

"""#### Number of Unique values"""

df.nunique()

"""## EDA

### Univariate Analysis
"""

df.describe()

"""#### Percentage of people who got promoted from each department"""

plt.rcParams['figure.figsize'] = [10, 5]
ct = pd.crosstab(df.department,df.is_promoted,normalize='index')
ct.plot.bar(stacked=True)
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))

"""While Technology department had highest percentage of employees getting promoted, Legal department has the least number. But we don't see major differences in terms of percentages.

#### Percentage of promotions across all the regions
"""

reg = pd.crosstab(df.region,df.is_promoted,normalize='index')
reg.plot.bar(stacked=True)
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))

"""#### Distribution of promotions among people with different Educational backgrounds"""

plt.rcParams['figure.figsize'] = [5, 5]
edu = pd.crosstab(df.education,df.is_promoted,normalize='index')
edu.plot.bar(stacked=True)
plt.rcParams['figure.figsize'] = [5, 5]
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))

"""#### Variation of promotion percentage with respect to gender"""

pd.crosstab(df.gender,df.is_promoted,normalize='index')

"""#### Difference in the percentage of promoted employees with respect to previous year ratings"""

rating = pd.crosstab(df.previous_year_rating,df.is_promoted,normalize='index')
rating.plot.bar(stacked=True)
plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))

"""#### Genderwise Pie Chart Depicting Promoted and Not promoted Employees"""

plt.style.use('seaborn')
plt.subplots(figsize=(10,6))
plt.subplot(1,2,1)
plt.pie(
    x=df[df.gender=='f'].is_promoted.value_counts(normalize=True),
    labels=['Not Promoted','Promoted'],
    explode=[0,0.2],
    autopct="%1.1f%%",
    shadow=True,
    textprops=dict(color='w',fontsize=14),
    colors=['#009999','#ff9933']
)
plt.title("Female")
plt.subplot(1,2,2)
plt.pie(
    x=df[df.gender=='m'].is_promoted.value_counts(normalize=True),
    labels=['Not Promoted','Promoted'],
    explode=[0,0.2],
    autopct="%1.1f%%",
    shadow=True,
    textprops=dict(color='w',fontsize=14),
    colors=['#009999','#ff9933']
)
plt.title("Male")
plt.legend(['Not Promoted','Promoted'],loc='upper right', bbox_to_anchor=(1, 0.5, 0.5, 0.5))

"""#### Promotion on the Basis of Length of Service

####
"""

plt.figure(figsize=(15,6))
sns.histplot(x='length_of_service',hue='is_promoted',data=df,palette='Set2',bins=30,kde=True);

"""### Multivariate Analysis

#### Average Training Score
"""

plt.figure(figsize=(15,6))
sns.kdeplot(x='avg_training_score',hue='is_promoted',data=df,shade=True);

"""#### Heatmap"""

plt.figure(figsize=(15,6))
correlation = df.corr(method='pearson')
sns.heatmap(correlation,annot=True);

"""## Data Modelling

#### Data Pre-Processing
"""

# Identify input and target columns
input_cols, target_col = df.columns[1:-1], df.columns[-1]
inputs_df, targets = df[input_cols].copy(), df[target_col].copy()

# Identify numeric and categorical columns
numeric_cols = df[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = df[input_cols].select_dtypes(include='object').columns.tolist()

# Impute and scale numeric columns
imputer = SimpleImputer().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = imputer.transform(inputs_df[numeric_cols])
scaler = MinMaxScaler().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])

"""#### One Hot Encode Categorical Columns"""

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(inputs_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names(categorical_cols))
inputs_df[encoded_cols] = encoder.transform(inputs_df[categorical_cols])

# Create training and validation sets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs_df[numeric_cols + encoded_cols], targets, test_size=0.25, random_state=42)
import warnings
warnings.filterwarnings('ignore')

"""### Random Forest"""

rf = RandomForestClassifier(random_state=42)
rf.fit(train_inputs,train_targets)

rf_train_score = rf.score(train_inputs,train_targets)
rf_val_score = rf.score(val_inputs,val_targets)
print('Random Forest Classifier Training Score: {}, Random Forest Classifier Validation score:{}'.format(rf_train_score,rf_val_score))

val_targets.value_counts() / len(val_targets)

"""## Conclusion

Training Accuracy is close to 100% and validation accuracy is 2% better than always predicting "No". It appears that model is overfitting as training sets are perfectly understood by model. However for other than training sets model is not able to predict.
"""