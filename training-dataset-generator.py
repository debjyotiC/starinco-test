from os import listdir
from os.path import isdir, join, splitext
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

dataset_path = "dataset/george_test_task"

all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]

filenames = []
y = []

print("Target classes:", all_targets)


def parse_image(path, target_size):
    image = Image.open(path)
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image


for index, target in enumerate(all_targets):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)

raw_images = []
labels = []

dropped, kept = 0, 0

for folder in range(len(all_targets)):
    all_files = join(dataset_path, all_targets[folder])
    for file_name in listdir(all_files):
        full_path = join(all_files, file_name)

        # Check if it's not .DS_Store and has .jpg extension
        if not file_name.startswith('.DS_Store') and splitext(file_name)[1] == '.jpg':
            print(full_path, folder)
            image_im = parse_image(full_path, (128, 128))

            if image_im.size == (128, 128):
                if image_im.mode != 'RGB':
                    image_im = image_im.convert('RGB')  # Convert to RGB mode
                image_array = img_to_array(image_im)

                raw_images.append(img_to_array(image_im))
                labels.append(folder + 1)

                print("Image Shape: ", image_im.size)
                kept += 1
            else:
                print('Image Dropped:', folder, image_im.size)
                dropped += 1

data_image_y = np.array(labels)

print(f"Kept {kept} files and dropped {dropped} in total of {dropped + kept}")

raw_images_array = np.array(raw_images)

np.savez('dataset/george_test_task.npz', out_x=raw_images_array, out_y=data_image_y)

print("saved NPZ file")
