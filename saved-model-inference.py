import numpy as np
from os import listdir
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

model_path = "saved-model/george_test_task"

# classify_image = "dataset/george_test_task/georges/0a74a65e38cf3682389f9780000e63b0.jpg"
classify_image = "dataset/george_test_task/non_georges/0a869d67deaaa70385fae7f70b92a557.jpg"

classes_values = listdir("dataset/george_test_task")


# Load the model
load_model = tf.keras.models.load_model(model_path)

# Load and preprocess the image
img = load_img(classify_image, target_size=(128, 128))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

results = load_model.predict(img_array, batch_size=1)
predicted_class_index = np.argmax(results)

predicted_class = classes_values[predicted_class_index]

print(f"Predicted Class: {predicted_class}")

