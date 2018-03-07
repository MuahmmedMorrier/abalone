import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv(r"newdata.csv", na_values='?', sep=',', index_col=None)

keys = df.keys()
# 7046
#12386
#19474
#33706 .022
for key in keys:
    if df[key].dtype == "object":
        df[key].fillna(df[key].mode()[0], inplace = True)
        le = LabelEncoder()
        le.fit(df[key])
        df[key] = le.transform(df[key])


X = df.iloc[:,0:8]
print(X)
Y = df.rings
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.33, random_state=33706)
clf = LinearRegression()
clf = clf.fit(x_train, y_train)
res = clf.predict(x_test)

print(str(2)+" RMSE = ", np.sqrt(mean_squared_error(y_test, res)))s