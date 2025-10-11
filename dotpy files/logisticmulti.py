import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)

digits.data[0]  

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])

plt.show()

digits.target[0:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

len(X_train)
len(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

model.score(X_test, y_test)
plt.matshow(digits.images[67])
plt.show()
digits.target[67]
model.predict([digits.data[67]])
model.predict(digits.data[0:5])

from sklearn.metrics import confusion_matrix
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
cm

import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
