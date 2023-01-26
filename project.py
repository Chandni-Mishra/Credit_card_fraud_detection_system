import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('creditcard.csv')

# Handling the Imbalance Dataset

# Segregating the data into 2 based on type of class
Real_data = data[data.Class == 0]
Fraud_data = data[data.Class == 1]
print(Real_data.shape)
print(Fraud_data.shape)

# There are only 492 data of fraud class available which probably 1 % of the whole dataset
# training with this dataset will not produce a high accuarcy.

# To handle that taking out the same amount of data from real class and concatinating it with fraud class will create
# a new dataset that has a pretty good distribution of both the classes 
new_Real_data = Real_data.sample(492)
print(new_Real_data.shape)

# Now lets concatinate the above data with fraud data sample
new_data = pd.concat([new_Real_data,Fraud_data])
print(data.shape)      ## Original dataset shape
print(new_Real_data.shape) ## new real dataset shape (which was segreggated from non-fraud class of data)
print(Fraud_data.shape)  ## Fraud dataset shape
print(new_data.shape)  #new dataset shape (created afer concatinating)

print(new_data.head())

print(new_data['Class'].value_counts())

# Above Distribution is pretty good 
# lets seperate the features and classes

x = new_data.drop(['Class'],axis = 1)
y = new_data['Class']

# spliting into test and train dataset for traning
# making use of sklearn's train and test split function

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape)
print(y.shape)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Now so our data has only 2 class so it is a binary classification problem
# A logistic Regression model can be best to use here

# defining a accuracy calculator function
def cal_accuracy(prediction,y):
    y = y.to_numpy()
    l = []
    for i in range(prediction.size):
        l.append(prediction[i] == y[i])
    accuracy = np.mean(l)
    return accuracy
    
model = LogisticRegression()
model.fit(X_train, Y_train)


# Calculating accuracy on train dataset
prediction = model.predict(X_train)
accuracy = cal_accuracy(prediction,Y_train)
print(f'Accuracy on train dataset is:- {accuracy * 100} %')

# Calculating accuracy on test dataset
prediction = model.predict(X_test)
accuracy = cal_accuracy(prediction,Y_test)
print(f'Accuracy on test dataset is:- {accuracy * 100} %')