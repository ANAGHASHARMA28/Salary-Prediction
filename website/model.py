import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
data = pd.read_csv('1614238459_salarydata.csv')
#Replacing missing '?' marks with mode of the attributes
data['workclass']=data['workclass'].replace('?','Private')
data['occupation']=data['occupation'].replace('?','Prof-specialty')
data['native-country']=data['native-country'].replace('?','United-States')
#Feature engineering
data['marital-status'].replace('Never-married', 'Not Married',inplace=True)
data['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
data['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
data['marital-status'].replace(['Married-spouse-absent'], 'Not Married',inplace=True)
data['marital-status'].replace(['Separated'], 'Separated',inplace=True)
data['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
data['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)

data['education'].replace('Preschool', 'school',inplace=True)
data['education'].replace('10th', 'HS-Grad',inplace=True)
data['education'].replace('11th', 'HS-Grad',inplace=True)
data['education'].replace('12th', 'HS-Grad',inplace=True)
data['education'].replace('1st-4th', 'school',inplace=True)
data['education'].replace('5th-6th', 'school',inplace=True)
data['education'].replace('7th-8th', 'school',inplace=True)
data['education'].replace('9th', 'HS-Grad',inplace=True)
data['education'].replace('HS-Grad', 'HS-Grad',inplace=True)
data['education'].replace('HS-grad', 'HS-Grad',inplace=True)
data['education'].replace('Some-college', 'College',inplace=True)
data['education'].replace('Assoc-acdm', 'College',inplace=True)
data['education'].replace('Assoc-voc', 'College',inplace=True)
data['education'].replace('Bachelors', 'Bachelors',inplace=True)
data['education'].replace('Masters', 'Masters',inplace=True)
data['education'].replace('Prof-school', 'Masters',inplace=True)
data['education'].replace('Doctorate', 'Doctorate',inplace=True)
#Label encoding
label_encoder = preprocessing.LabelEncoder()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in ['workclass', 'education', 'education-num', 'marital-status',
       'occupation', 'relationship', 'race', 'sex','native-country', 'salary']:
    data[i] = label_encoder.fit_transform(data[i])

X = data.drop(['salary'], axis = 1)
y = data['salary']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

#Scaling the model
standardisation = preprocessing.StandardScaler()
X = standardisation.fit_transform(X)
X = pd.DataFrame(X)

from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
m=gb.fit(X_train,y_train.values.ravel())
predictions=gb.predict(X_test)

pickle.dump(m, open('model.pkl', 'wb'))

