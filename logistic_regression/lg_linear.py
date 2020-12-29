import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load data
dataset = load_iris()
X, Y, names = dataset['data'], dataset['target'], dataset['target_names']
target = dict(zip(np.array([0, 1, 2]), names))

# devide to training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# normalize train data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_pr = scaler.transform(X_train)

# logistic regression
model = LogisticRegression()
model.fit(X_train_pr, Y_train)

# normalize test data (with train scalar)
X_test_pr = scaler.transform(X_test)

# run model on test
Y_train_pred, Y_test_pred = model.predict(X_train_pr), model.predict(X_test_pr) 

# print results
print('accuracy on train data: ', accuracy_score(Y_train, Y_train_pred))
print('accuracy on test data: ', accuracy_score(Y_test, Y_test_pred))

'''
print(classification_report(Y_test, Y_test_pred))
print(confusion_matrix(Y_test, Y_test_pred))
'''

# test 20 random observations
indices = np.random.randint(150, size=20)
X_pred, Y_true = X[indices], Y[indices]

X_pred_pr = scaler.transform(X_pred)

Y_pred = model.predict(X_pred_pr)

target_true, target_pred = [], []
cnt = 0.0
for i in range(len(Y_true)):
    t = Y_true[i]
    p = Y_pred[i]
    if t != p:
        cnt += 1
    target_true.append(target[t])
    target_pred.append(target[p])

print('accuracy on 20 random observations: ', 1 - (cnt / 20))