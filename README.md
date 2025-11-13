# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: AAKIL AHAMED S
RegisterNumber:  212224040002

```

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics  


data = pd.read_csv("Salary.csv")


data.head()
data.info()
data.isnull().sum()


le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()


x = data[["Position", "Level"]]
x.head()
y = data["Salary"]
y.head()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)


y_pred = dt.predict(x_test)
y_pred


r2 = metrics.r2_score(y_test, y_pred)


print("R2 Score: ",r2)
```

## Output:

<img width="571" height="308" alt="Screenshot 2025-10-06 105102" src="https://github.com/user-attachments/assets/18f8a484-9f2e-46a4-bcb5-72dda8879e81" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
