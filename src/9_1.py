import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.models.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

n_train = 10000
X_train = np.random.rand(n_train, 2) - 0.5  # -0.5 <= x1 <= 0.5, -0.5 <= x2 <= 0.5
y_train = (16 * X_train[:, 0] ** 2 + X_train[:, 0] * X_train[:, 1] + 8 * X_train[:, 1] ** 2 - X_train[:, 0] - X_train[:, 1] + np.log(1 + X_train[:, 0] ** 2 + X_train[:, 1] ** 2))
y_train = y_train.reshape(n_train, 1)

n_val = n_train // 5
X_val= np.random.rand(n_val, 2) - 0.5  # -0.5 <= x1 <= 0.5, -0.5 <= x2 <= 0.5
y_val = (16 * X_val[:, 0] ** 2 + X_val[:, 0] * X_val[:, 1] + 8 * X_val[:, 1] ** 2 - X_val[:, 0] - X_val[:, 1] + np.log(1 + X_val[:, 0] ** 2 + X_val[:, 1] ** 2))
y_val = y_val.reshape(n_val, 1)

history = model.fit(X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_val, y_val))

n_test = n_train // 5
X_test= np.random.rand(n_test, 2) - 0.5  # -0.5 <= x1 <= 0.5, -0.5 <= x2 <= 0.5
y_test = (16 * X_test[:, 0] ** 2 + X_test[:, 0] * X_test[:, 1] + 8 * X_test[:, 1] ** 2 - X_test[:, 0] - X_test[:, 1] + np.log(1 + X_test[:, 0] ** 2 + X_test[:, 1] ** 2))
y_test = y_test.reshape(n_test, 1)

test_loss = model.evaluate(X_test, y_test)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.legend()
plt.show()


print("Test Loss:", test_loss)
