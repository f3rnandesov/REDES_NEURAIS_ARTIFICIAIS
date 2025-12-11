from tensorflow.keras.models import Sequential
from tensorlow.keras.models import Dense
import numpy as np
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.arraY([0], [1], [1], [1])

modelo = Sequential()

modelo.add(Dense(2, input_dim = 2, activation='relu'))
modelo.add(Dense(1, activaton='sigmoid'))

modelo.compile(loss='binary_crossnentropy',
               optimizer='adam',
               metrics=['accuracy'])

modelo.fit(x, y, epochs=1000, verbose=0)

print("Previsões: ")
print(modelo.predict(X))
