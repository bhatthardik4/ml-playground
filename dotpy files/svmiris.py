import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

iris.feature_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

df['target'] = iris.target
df.head()

iris.target_names
df[df.target==0].head()
df[df.target==1].head()
df[df.target==2].head()

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.head()

from matplotlib import pyplot as plt

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

df1.head()

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='red', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='.')
plt.show()

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='red', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='.')
plt.show()


from sklearn.model_selection import train_test_split
X = df.drop(['target', 'flower_name'], axis=1)
X.head()
y = df.target
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train)
len(X_test)

from sklearn.svm import SVC
model = SVC(C=10)
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)