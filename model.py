# import neccesary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import seaborn as sns

# load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# check dataset shapes 
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# normalize data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# reshape data to include channel dimensions for ConvNets
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# use matplotlib to see data
index = 0
plt.imshow(X_train[index], cmap=plt.cm.binary)
plt.title(f"Label: {y_train[index]}")
plt.show()
print(y_train[index])

# nueral network model (changed to cnn from feed forward)
model = keras.Sequential([
    # Conv layer 1
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    # Conv layer 2
    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    # Conv layer 3
    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    # Flatten layer 
    keras.layers.Flatten(),
    # Fully connected layer
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    # Output layer
    keras.layers.Dense(10, activation='softmax')
])

# print model summary
model.summary()

# train on dataset
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(X_train, y_train, epochs = 5, validation_data = (X_test, y_test))

# evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict labels
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# confusion matrix to check for which numbers are more likely to be mislabeled 
confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)

# visualize confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# save the trained model
model.save('/Users/keerthanajayamoorthy/Desktop/Digit Classifier Project/model.h5')

