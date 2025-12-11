import keras
import tensorflow

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train=X_train/255
X_test=X_test/255

X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

convolutional_neural_network=keras.models.Sequential([
keras.layers.Conv2D(filters=25,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
keras.layers.MaxPooling2D((2,2)),
keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
keras.layers.MaxPooling2D((2,2)),
keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
keras.layers.MaxPooling2D((2,2)),
keras.layers.Flatten(),
keras.layers.Dense(64,activation='relu'),
keras.layers.Dense(10,activation='softmax')]
)

convolutional_neural_network.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
convolutional_neural_network.fit(X_train, y_train, epochs=5)
convolutional_neural_network.evaluate(X_test,y_test)
y_predicted_by_model=convolutional_neural_network.predict(X_test)

import numpy as np

np.argmax(y_predicted_by_model[0])
y_predicted_by_model[0]
