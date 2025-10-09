import pandas as pd
df = pd.read_csv('data/carprices (1).csv')
df

import matplotlib.pyplot as plt
plt.scatter(df['Mileage'], df['Sell Price($)']) 

X = df[['Mileage','Age(yrs)']]
y =  df[['Sell Price($)']]# Scatter plot to visualize the relationship

X

y

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

len(X_train)
len(X_test)

X_train

from sklearn.linear_model import LinearRegression
clf = LinearRegression()

clf.fit(X_train, y_train)
clf.predict(X_test)

y_test
clf.score(X_test, y_test)