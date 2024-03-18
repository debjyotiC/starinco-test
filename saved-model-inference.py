import numpy as np
from os import listdir
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

model_path = "saved-model/george_test_task"

classify_image = "dataset/george_test_task/georges/0a5f7b5996063605dd05887ef4d31855.jpg"

classes_values = listdir("dataset/george_test_task")
classes_values.remove('.DS_Store')

# Load the model
load_model = tf.keras.models.load_model(model_path)

# Load and preprocess the image
img = load_img(classify_image, target_size=(128, 128))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
image_data = img_array


results = load_model.predict(image_data, batch_size=1)
predicted_class_index = np.argmax(results)

predicted_class = classes_values[predicted_class_index]

print(f"Predicted Class: {predicted_class}")

