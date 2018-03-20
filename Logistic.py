import pandas as pd
import numpy as np
import matplotlib.pyplot as mp

dataset=pd.read_csv('/home/varun/Downloads/Machine Learning A-z/Logistic_Regression/Logistic_Regression/Social_Network_Ads.csv')
X=dataset.iloc[: ,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.25)

from sklearn.preprocessing import  StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

from sklearn.linear_model import  LogisticRegression
regressor=LogisticRegression(random_state=0)
regressor.fit(X_train,y_train)
predic=regressor.predict(X_test)
print(predic)

from sklearn.metrics import  confusion_matrix
cm=confusion_matrix(y_test,predic)
print(cm)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
mp.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'yellow')))
mp.xlim(X1.min(), X1.max())
mp.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    mp.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
mp.title('Training Set')
mp.xlabel('Age')
mp.ylabel('Salary')
mp.legend()
mp.show()

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
mp.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
mp.xlim(X1.min(), X1.max())
mp.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    mp.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
mp.title('Test Set')
mp.xlabel('Age')
mp.ylabel('Salary')
mp.legend()
mp.show() 