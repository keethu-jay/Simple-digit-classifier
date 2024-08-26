# import neccesary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import seaborn as sn

# load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# check dataset shapes 
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# normalize data
X_train = X_train / 255
X_test = X_test / 255

# use matplotlib to see data
index = 0
plt.imshow(X_train[index], cmap=plt.cm.binary)
plt.title(f"Label: {y_train[index]}")
plt.show()
print(y_train[index])

# flatten the data for the dense layers
X_train_flat = X_train.reshape(len(X_train), (28 * 28))
X_test_flat = X_test.reshape(len(X_test), (28 * 28))

print(X_train_flat.shape)

# nueral network model
model = keras.Sequential([
    # layer 1
    keras.layers.Dense(128, input_shape=(784,), activation = 'relu'),
    # layer 2
    keras.layers.Dense(64, activation = 'sigmoid'),
    # layer 3
    keras.layers.Dense(12, activation = 'sigmoid'),
    # layer 4
    keras.layers.Dense(10, activation = 'softmax')
])

# print model summary
model.summary()

# train on dataset
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_flat, y_train, epochs = 5)

# save the trained model
model.save('/Users/keerthanajayamoorthy/Desktop/Digit Classifier Project/model.h5')

# test 
model.evaluate(X_test_flat, y_test)

# store models prediction 

y_pred = model.predict(X_test_flat)
y_pred_labels = [np.argmax(i) for i in y_pred]

# confusion matrix to check for which numbers are more likely to be mislabeled 
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)
sn.heatmap(confusion_matrix, annot=True, fmt='d')
print(confusion_matrix)


