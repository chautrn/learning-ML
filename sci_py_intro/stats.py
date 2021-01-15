from pandas import read_csv

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape
print("\nShape:")
print(dataset.shape)

# head
print("\nFirst 20:\n")
print(dataset.head(20))

# description
print("\nDescription:")
print(dataset.describe())

# class distribution
print("\nDistribution:")
print(dataset.groupby('class').size())
