import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math

def sigmoid(X,y,w):
    scores = np.dot(X,w)
    return np.exp(y * scores) / (1 + np.exp(scores))

def alpha(X,w):
    scores = np.dot(X,w)
    return np.exp(scores) / (1 + np.exp(scores))

def normal(X,w):
    scores = np.dot(X,w)
    return 1 / (1 + np.exp(scores))

def log_likelihood(X, y, w):
    scores = np.dot(X,w.T)
    ll = np.sum( y*scores - np.log(1 + np.exp(scores) ) )
    return ll

def newton(X,w,y,lam=False):
    a = sigmoid(X,y,w)
    # a = alpha(X,w)
    # a = normal(X,w)

    g = X.T.dot(y-a)

    R = np.diag( a * ( 1 - a ) )


    H = - X.T.dot(R).dot(X)

    if lam:
        hessian_inv = np.linalg.inv(H + lam*np.eye(w.shape[0]))
    else:
        hessian_inv = np.linalg.inv(H)

    return (hessian_inv @ g)

def logistic_regression(X,y,num_steps):

    w = np.zeros(X.shape[1])
    batches = math.floor(X.shape[0] / 2000)

    for step in range(num_steps):

        for batch in range(batches):        

            batch_X = X[2000*batch:2000*(batch+1)]
            batch_y = y[2000*batch:2000*(batch+1)]

            w = w - step_size * newton(batch_X,w,batch_y,lam=0.1e-5)

        if step % 1 == 0:
            print(log_likelihood(X,y,w))

    return w




df = pd.read_csv('norm_dataset.csv')
df2 = pd.read_csv('norm_test_dataset.csv')

X = np.array(df.values[:,1:])
y = np.array(df.values[:,0])

X_test = np.array(df2.values[:,1:])
y_test = np.array(df2.values[:,0])

step_size = 0.03
# step_size = 1.00


w = logistic_regression(X,y,num_steps=50)

# sklearn_lr(X,y)
clf = LogisticRegression(fit_intercept=True,C=1e15)
clf.fit(X,y)

# print(clf.intercept_)
# print(clf.coef_)

preds = np.round(sigmoid(X,y,w))

# print((preds == y).sum().astype(float))
# print(len(preds))

# print('Accuracy: {0}' .format( (preds == y).sum().astype(float) / len(preds) ))
# print('Sklearn accuracy: {0}' .format( clf.score(X,y)) )


preds = np.round(sigmoid(X_test,y_test,w))

print('Accuracy: {0}' .format( (preds == y_test).sum().astype(float) / len(preds) ))
print('Sklearn accuracy: {0}' .format( clf.score(X_test,y_test)) )
