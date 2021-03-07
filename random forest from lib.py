import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


df = pd.read_csv('C:/Users/jkrueger/Downloads/petrol_consumption.csv')

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

mean_abs_loss_arr = []
mean_squared_loss_arr = []
rmse_loss_arr = []


for i in range(1, 50):
    regressor = RandomForestRegressor(n_estimators=i, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    mean_abs = metrics.mean_absolute_error(y_test, y_pred)
    mean_squared = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    mean_abs_loss_arr.append(mean_abs)
    mean_squared_loss_arr.append(mean_squared)
    rmse_loss_arr.append(rmse)
    
    if i == 1:
        pass
    else:
        rmse_percent_difference = (abs(rmse_loss_arr[i-1] - rmse_loss_arr[i-2])) \
                            / (rmse_loss_arr[i-1] + rmse_loss_arr[i-2])
        if rmse_percent_difference < .001:
            print(i, rmse_percent_difference)
            break



