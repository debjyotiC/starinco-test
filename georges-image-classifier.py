import numpy as np
import tensorflow as tf
from os import listdir
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

classes_values = listdir("dataset/george_test_task")
classes_values.remove('.DS_Store')

classes = len(classes_values)

print("Target classes: ", classes_values)

thermal_data = np.load("dataset/george_test_task.npz", allow_pickle=True)
x_data, y_data = thermal_data['out_x'], thermal_data['out_y']

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)

train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))


input_shape = x_train[0].shape

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(classes, activation='softmax')
])


model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# this controls the batch size
BATCH_SIZE = 25
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

model.save("saved-model/george_test_task")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Training Accuracy: {round(np.average(acc), 3)}")
print(f"Validation Accuracy: {round(np.average(val_acc), 3)}")

predicted_labels = model.predict(x_test)
actual_labels = y_test

np.savez('test-results/george_test_task.npz', out_x=predicted_labels, out_y=actual_labels)

epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(2, 1)

# plot loss
axs[0].plot(epochs, loss, '-', label='Training loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')
# plot accuracy
axs[1].plot(epochs, acc, '-', label='Training acc')
axs[1].plot(epochs, val_acc, 'b', label='Validation acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend(loc='best')
plt.show()
