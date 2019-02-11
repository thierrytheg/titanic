#https://www.kaggle.com/chapagain/titanic-solution-a-beginner-s-guide/notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
#sns.set()

sys.__stdout__=sys.stdout
os.chdir(r'C:\Users\Thierry\Desktop\titanic')

#identify train & test sets
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#OVERVIEW OF TRAINING SET

#display first 5 rows of training dataset
#print(train.head())

#return size of dataset
#print(train.shape)

#return dataset characteristics
#print(train.describe())

#return dataset characteristics for objects
#print(train.describe(include=['O']))

#return variables and format
#print(train.info())

#return summary of variables with missing or incomplete data
#print(train.isnull().sum())

#OVERVIEW OF TESTING SET

#display first 5 rows of training dataset
#print(test.head())

#return size of dataset
#print(test.shape)

#return dataset characteristics
#print(test.describe())

#return dataset characteristics for objects
#print(test.describe(include=['O']))

#return variables and format
#print(test.info())

#return summary of variables with missing or incomplete data
#print(test.isnull().sum())

#RELATIONSHIP BETWEEN VARIABLES AND SURVIVAL

survived=train[train['Survived']==1]
not_survived=train[train['Survived']==0]

#print("Survived: %d (%.1f%%)" %(len(survived),(len(survived)/len(train))*100))
#print("Not Survived: %d (%.1f%%)" %(len(not_survived),(len(not_survived)/len(train))*100))
#print("Total: %d" %len(train))

#higher class passengers have better chance of survival
#print(train.Pclass.value_counts())

#group by survivals and deaths by class
#print(train.groupby('Pclass').Survived.value_counts())

#print(train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean())

#sns.barplot(x='Pclass', y='Survived', data=train)
#plt.show()

#SEX VS SURVIVAL

#PCLASS & SEX VS SURVIVAL

#PCLASS, SEX & EMBARKED VS SURVIVAL

#EMBARKED VS SURVIVAL

#PARCH VS SURVIVAL

#SIBSP VS SURVIVAL

#AGE VS SURVIVAL

#CORRELATING FEATURES

#plt.figure(figsize=(15,6))
#sns.heatmap(train.drop('PassengerId',axis=1).corr(),vmax=1,square=True, annot=True)
#plt.show()

#FEATURE EXTRACTION

#combine train and test
train_test_data=[train,test]

#insert title column in both training and testing sets
for dataset in train_test_data:
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.')
#print(train.head())

#gender distribution by title
#print(pd.crosstab(train['Title'],train['Sex']))

#replace less common titles with "Other"

for dataset in train_test_data:
    dataset['Title']=dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Other')
    dataset['Title']=dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title']=dataset['Title'].replace('Ms', 'Miss')
    dataset['Title']=dataset['Title'].replace('Mme', 'Mrs')

#print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())

#convert titles into numeric form
title_mapping={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title'] =dataset['Title'].fillna(0)

#print(train.head())

#convert sex into numeric form
sex_mapping={'female': 1, 'male': 0}
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map(sex_mapping).astype(int)
#print(train.head())

#EMBARKED FEATURE

#return unique values for variable embarqued
#print(train.Embarked.unique())

#number passengers for each embarked category
#print(train.Embarked.value_counts())

#replace NAN with S
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
#print(train.head())

#convert embarking port into numberic
embarked_mapping={'S': 0, 'C': 1, 'Q': 2} 
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)
#print(train.head())

#Age feature
#create age ranges
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)

#print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

#map age according to ageband
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

#print(train.head())

#FARE FEATURE

#replace missing fare features with median of fare
for dataset in train_test_data:
    dataset['Fare']=dataset['Fare'].fillna(train['Fare'].median())

#createfareband
train['FareBand'] = pd.qcut(train['Fare'], 4)
#print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
    
#map_fare_according to fareband
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#combine sibsp & parch features to create family size variable
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

#print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

#create alone variable
for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
#print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

#drop unncecessary columns
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

#CLASSIFICATION & ACCURACY

#define training and testing sets
X_train=train.drop('Survived',axis=1)
y_train=train['Survived']
X_test=test.drop('PassengerId',axis=1).copy()

#print(X_train.shape, y_train.shape,X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

#logistic regression
clf=LogisticRegression()
clf.fit(X_train,y_train)
y_pred_log_reg=clf.predict(X_test)
acc_log_reg=round(clf.score(X_train,y_train)*100,2)

#svm
clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)

#linear svm
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)

#knearest_neighbors
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)

#decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)

#random forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)

#gaussian naive bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)

#perceptron
clf = Perceptron(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_perceptron = clf.predict(X_test)
acc_perceptron = round(clf.score(X_train, y_train) * 100, 2)

#stochastic gradient descent (SGD)
clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)

#rank prediction models

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_perceptron, acc_sgd]
    })

print(models.sort_values(by='Score', ascending=False))


#confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')
cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)
plt.show()

#final_submission
submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred_random_forest})
submission.to_csv('submission.csv',index=False)
