import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

model = keras.models.Sequential([
    keras.Input(shape=(22,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

data = pd.read_csv('../assets/parkinsons.csv')

data = data.drop(columns=['name'])

label = data[['status']]

data = data.drop(columns=["status"])

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42, stratify=label)

history = model.fit(X_train, y_train, epochs=250, batch_size=1, validation_split=0.1)

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
