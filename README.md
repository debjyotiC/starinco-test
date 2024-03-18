## St. George Image Classifier

Welcome to the St. George Image Classifier! This project aims to identify St. George in images using a simple Convolutional Neural Network (CNN) based approach.

### Instructions

Follow these steps to get started with the classifier:

#### Step 1: Install Dependencies

Install all required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

#### Step 2: Download Images

Run the `download-image.py` script to download the images provided by the two CSV files in the `image_link` folder:

```bash
python3 download-image.py
```

#### Step 3: Generate Training Dataset

Generate NPZ files of the processed images by running the `training-dataset-generator.py` script:

```bash
python3 training-dataset-generator.py
```

#### Step 4: Train the Classifier

Run the `georges-image-classifier.py` script to train the classifier. This step will save two important files: the trained model and unseen metrics file:

```bash
python3 georges-image-classifier.py
```

#### Step 5: Evaluate Performance on Unseen Data

Visualize the classifier's performance on unseen data by running the `unseen-inference-metrics.py` script. This will display accuracy, precision, and F-score along with a confusion matrix:

```bash
python3 unseen-inference-metrics.py
```

### Model Metrics

The current model based on a simple CNN exhibits the following metrics:

![Confusion Matrix](https://github.com/debjyotiC/starinco-test/blob/main/images/confusion_matrix.png)

