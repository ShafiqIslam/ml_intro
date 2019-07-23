from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# print(type(iris))
# print(iris)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=109)

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
