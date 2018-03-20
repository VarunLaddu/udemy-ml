import matplotlib.pyplot as mp
import numpy as np
import pandas as pd

dd=pd.read_csv('/home/varun/Position_Salaries.csv')
X=dd.iloc[:,1:2].values
y=dd.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=637,random_state=0)
reg.fit(X,y)
print(reg.predict(6.5))
X_grid=np.arange(min(X),max(X),0.001)
X_grid=X_grid.reshape(len(X_grid),1)


mp.plot(X_grid,reg.predict(X_grid),color='green')
mp.show()