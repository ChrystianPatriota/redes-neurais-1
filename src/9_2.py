import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

model = keras.models.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

n_train = 15000
cycle = 2

n_train //= 2
theta_train = 2 * cycle * np.pi * np.random.rand(n_train, 1)
X_train_0, y_train_0 = np.array([theta_train/4*np.cos(theta_train), theta_train/4*np.sin(theta_train)]).transpose().reshape((n_train,2)), np.zeros((n_train, 1))
X_train_1, y_train_1 = np.array([(theta_train/4+0.8)*np.cos(theta_train), (theta_train/4+0.8)*np.sin(theta_train)]).transpose().reshape((n_train,2)), np.ones((n_train, 1))
X_train = np.concatenate((X_train_0, X_train_1), axis=0)
y_train = np.concatenate((y_train_0, y_train_1), axis=0)

n_val = n_train // 5
theta_val = 2 * cycle * np.pi * np.random.rand(n_val, 1)
X_val_0, y_val_0 = np.array([theta_val/4*np.cos(theta_val), theta_val/4*np.sin(theta_val)]).transpose().reshape((n_val,2)), np.zeros((n_val, 1))
X_val_1, y_val_1 = np.array([(theta_val/4+0.8)*np.cos(theta_val), (theta_val/4+0.8)*np.sin(theta_val)]).transpose().reshape((n_val,2)), np.ones((n_val, 1))
X_val = np.concatenate((X_val_0, X_val_1), axis=0)
y_val = np.concatenate((y_val_0, y_val_1), axis=0)

history = model.fit(X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_val, y_val))

n_test = n_train // 5
theta_test = 2 * cycle * np.pi * np.random.rand(n_test, 1)
X_test_0, y_test_0 = np.array([theta_test/4*np.cos(theta_test), theta_test/4*np.sin(theta_test)]).transpose().reshape((n_test,2)), np.zeros((n_test, 1))
X_test_1, y_test_1 = np.array([(theta_test/4+0.8)*np.cos(theta_test), (theta_test/4+0.8)*np.sin(theta_test)]).transpose().reshape((n_test,2)), np.ones((n_test, 1))
X_test = np.concatenate((X_test_0, X_test_1), axis=0)
y_test = np.concatenate((y_test_0, y_test_1), axis=0)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()