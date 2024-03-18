from os import listdir
from os.path import join, basename
import requests
import pandas as pd

image_link_path = "image_links"
download_dir = "dataset/george_test_task"

all_targets = listdir(image_link_path)

for csv_file in all_targets:
    full_path = join(image_link_path, csv_file)
    print(f"Processing image links from {csv_file}")
    df = pd.read_csv(full_path)

    image_folder_name = csv_file.split(".")[0]

    image_urls = df.iloc[:, 0]

    for image_url in image_urls:
        response = requests.get(image_url)

        if response.status_code == 200:
            print(f"Got image from url {image_url}")
            filename = basename(image_url)
            with open(join(download_dir, image_folder_name, filename), 'wb') as img_file:
                img_file.write(response.content)
