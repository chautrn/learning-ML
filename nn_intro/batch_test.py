from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into input (x) and output (y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for i in range(1, 16):
    model.fit(X, Y, epochs=150, batch_size=i*10, verbose=0) 
    _, accuracy = model.evaluate(X, Y, verbose=0)
    print('Batch Size: %d Accuracy: %.2f' % (i*10, accuracy*100))
