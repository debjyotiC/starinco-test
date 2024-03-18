# St. George image classifier
A simple CNN based image classifier that can identify St. George in an image 
# Step:1 
Install all dependencies using pip.
```bash
pip install -r requirements.txt
```
# Step:2
Run the 'download-image.py' file to download the images provided by the two csv files in 'image_link' folder.
```bash
python3 download-image.py
```
# Step:3
Run the 'training-dataset-generator.py' to generate npz files of the processed images.
```bash
python3 training-dataset-generator.py
```
# Step:4
Run the 'georges-image-classifier.py' to classify the processed images.
```bash
python3 georges-image-classifier.py
```
This code will save two important files i) the model and ii) unseen metrics file.

# Step:5
To visualise the classifier's performance on unseen data, run 'unseen-inference-metrics.py' file. 
This code will show accuracy, precision and f-score along with a confusion matrix.
```bash
python3 unseen-inference-metrics.py
```
