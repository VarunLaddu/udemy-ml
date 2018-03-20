import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
from clyent import color

dd=pd.read_csv('Position_salaries.csv')
X=dd.iloc[:,1:2].values
y=dd.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
reg= DecisionTreeRegressor(random_state=0)
reg.fit(X,y)
y_pred=reg.predict(6.5)
print(y_pred)

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)

mp.scatter(X,y)
mp.plot(X_grid,reg.predict(X_grid),color='green')
mp.show()