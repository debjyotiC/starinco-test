import os
import pandas as pd
import requests

image_link_path = "image_links"
download_dir = "dataset/george_test_task"
saved_model_dir = "saved-model"
test_result_dir = "test-results"
performance_images = "images"

# Create necessary directories if they don't exist
for directory in [download_dir, saved_model_dir, test_result_dir, performance_images]:
    os.makedirs(directory, exist_ok=True)

all_targets = os.listdir(image_link_path)

for csv_file in all_targets:
    full_path = os.path.join(image_link_path, csv_file)
    print(f"Processing image links from {csv_file}")
    df = pd.read_csv(full_path)

    image_folder_name = csv_file.split(".")[0]

    image_folder_path = os.path.join(download_dir, image_folder_name)

    os.makedirs(image_folder_path, exist_ok=True)

    image_urls = df.iloc[:, 0]

    for image_url in image_urls:
        response = requests.get(image_url)

        if response.status_code == 200:
            print(f"Got image from url {image_url}")
            filename = os.path.basename(image_url)
            with open(os.path.join(download_dir, image_folder_name, filename), 'wb') as img_file:
                img_file.write(response.content)
