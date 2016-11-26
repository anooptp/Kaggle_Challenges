# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 06:37:53 2016

@author: anooptp
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

titanic = pd.read_csv("train.csv")
print(titanic.head())

print(titanic.describe())

#print(titanic["Age"])
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#print(titanic["Age"].median())

print(titanic.describe())

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

print(titanic.describe())

print(titanic["Sex"].unique())
print(titanic["Embarked"].unique())

# The most common embarkation port is S, so let's assume everyone got on there.
# Replace all the missing values in the Embarked column with S.
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# We'll assign the code 0 to S, 1 to C and 2 to Q.
titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 1
titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 2

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()

kf =KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []

for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
#print(predictions)

import numpy as np
# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
#print(predictions)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

#print(predictions)
accuracy = metrics.accuracy_score(predictions, titanic["Survived"])
print (accuracy)

print(len(list(filter(lambda x: x==True, map(lambda x,y:x==y, predictions, titanic["Survived"])))) / len(predictions))

logreg = LogisticRegression()
scores = cross_val_score(logreg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())


titanic_test = pd.read_csv("test.csv")
#print(titanic_test.describe())
#titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
#print(titanic_test.describe())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
#print(titanic_test.describe())

# Replace all the missing values in the Embarked column with S.
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

# We'll assign the code 0 to S, 1 to C and 2 to Q.
titanic_test.loc[titanic_test["Embarked"] == 'S', "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == 'C', "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == 'Q', "Embarked"] = 2
#print(titanic_test.describe())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
print(titanic_test.describe())

logreg = LogisticRegression()
logreg.fit(titanic[predictors], titanic["Survived"])
predictions = logreg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
})

#print(submission)

submission.to_csv("output.csv", index=False, cols =('PassengerId', 'Survived') )
