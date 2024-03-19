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

<div style="text-align:center;">
    <img alt="Confusion Matrix" src="https://github.com/debjyotiC/starinco-test/blob/main/images/confusion_matrix.png" style="display:block; margin:auto; width:400px; height:350px;">
</div>

### Rationale behind Model Selection and Possible Improvements

#### Model Selection Rationale:
1. **Convolutional Neural Network (CNN)**:
   - The architecture of the chosen CNN has been kept simple (i.e. 2 layers) so that training it does not explicitly require a specialised hardware like GPU.
   - Drop out layers have been added to prevent over fitting.

2. **Layer Configuration**:
   - The model starts with a sequence of convolutional layers with increasing filters, allowing the network to capture complex patterns.
   - Max-pooling layers are employed to down sample the feature maps, aiding in spatial feature extraction while reducing computational complexity.
   - A dense layer with ReLU activation functions is used to capture higher-level abstractions in the flattened feature map.
   - The final dense layer with softmax activation performs multi-class classification.

#### Possible Improvements:
1. **Data Augmentation**:
   - Augmenting the image data could help in creating additional training samples, reducing over fitting and improving generalization.
   - Techniques like rotation, flipping, and scaling can be applied to increase the diversity of the training set.

2. **Hyperparameter Tuning**:
   - Experimenting with different learning rates, batch sizes, and optimizer configurations may lead to better convergence and performance.
   - Utilizing learning rate schedules or adaptive learning rate algorithms like AdamW could potentially enhance training stability and speed.

3. **Model Architecture Modifications**:
   - Increasing the depth or width of the CNN architecture might capture more intricate features.
   - Even moving to more complex model like ResNet50 or MobileNet could prove benificial. 
   - Adding batch normalization layers could mitigate over fitting.

4. **Ensemble Methods**:
   - Building an ensemble of models by training multiple CNNs with different initializations or architectures could potentially improve overall performance.
   - Combining predictions from multiple models can often yield better results than a single model.

### Conclusion:
In this task, a basic CNN-based model was trained for image classification with 78% accuracy. The said model was chosen owning its simplicity, the dataset made available for the training had images with extensions other then ".jpg", such images were not processed during the trainig. Moreover, majority of the images provided were of different height and weidth, this was tacked using python's PIL package. The model initially had more confidence on the training data compared to the testing set, this was tacked using dopout layers. Overall, the simple CNN model is nowhere near perfect, instead it proves as a good staring point into duilding networks for image classification. 

