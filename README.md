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

<img alt="Confusion Matrix" height="300" src="https://github.com/debjyotiC/starinco-test/blob/main/images/confusion_matrix.png" width="300"/>

### Rationale behind Model Selection and Possible Improvements

#### Model Selection Rationale:
1. **Convolutional Neural Network (CNN)**:
   - CNNs are widely used for image classification tasks due to their ability to automatically learn spatial hierarchies of features.
   - The architecture of the chosen CNN has been kept simple so that training it does not explicitly require a specialised hardware like GPU.

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
   - Adding batch normalization layers could mitigate over fitting.

4. **Ensemble Methods**:
   - Building an ensemble of models by training multiple CNNs with different initializations or architectures could potentially improve overall performance.
   - Combining predictions from multiple models can often yield better results than a single model.

### Conclusion:
In this task, a basic CNN-based model was trained for image classification with 77% accuracy. 
The selected architecture demonstrated the capability to learn relevant features from the data, resulting in decent accuracy on both training and validation sets.
Through techniques like data augmentation, hyperparameter tuning, and potential modifications in the model architecture, the classification performance could be further enhanced. 
Moreover, experimenting with ensemble methods could provide additional boosts in accuracy.
