# compare algorithms
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values

# X is all the inputs, sepal length, width, petal length, width etc.
X = array[:,0:4]

# y is the output, the resulting flower type from those inputs
Y = array[:,4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=1)

# make predictions
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# evaluate predictions
print('\n Accuracy:')
print(accuracy_score(Y_validation, predictions))
print('\n Confusion Matrix:')
print(confusion_matrix(Y_validation, predictions))
print('\n Classification Report:')
print(classification_report(Y_validation, predictions))
