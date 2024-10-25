# Multi-Class Text Classification Notebook

This notebook demonstrates a multi-class text classification pipeline, primarily focused on processing and classifying textual data using PyTorch and Hugging Face's Transformers. The project involves loading data, preprocessing, model configuration, and evaluation.

## Project Structure and Key Components

1. **Dependencies**
   - Installs essential libraries like `matplotlib`, `pandas`, `numpy`, `torch`, and `transformers` for data handling, model training, and performance visualization.

2. **Environment and Device Setup**
   - Configures the notebook to leverage GPU if available for faster model training:
     ```python
     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
     os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2"
     ```
   - Initializes the `torch` device and sets random seeds for reproducibility.

3. **Data Loading and Preprocessing**
   - Loads JSON data files from the specified directory and processes them into a pandas DataFrame.
   - `multi_label_to_one_hot` function:
     - Converts multi-class labels into a one-hot encoding format.
     - Parameters:
       - `labels`: List of class labels for each sample.
       - `num_classes`: Total number of distinct classes.
     - Returns: A numpy array of one-hot encoded labels.

4. **Model Architecture**
   - Utilizes a transformer model (`AutoModelForSequenceClassification`) for multi-class classification.
   - Tokenizes input data using `AutoTokenizer` to prepare for model input.

5. **Custom Dataset and Dataloader**
   - Implements a PyTorch `Dataset` class to handle text and label data:
     - Processes data into tensors suitable for model input.
   - Creates DataLoader objects for batch training and validation.

6. **Training and Evaluation**
   - Configures training loop with optimizer (`AdamW`) and handles forward and backward propagation.
   - Tracks model accuracy on the validation set using `accuracy_score` from `sklearn.metrics`.
   - Evaluation metrics:
     - Accuracy is computed post-training to gauge model performance.

7. **Results and Visualization**
   - Provides mechanisms to visualize accuracy and loss over epochs using `matplotlib`.

## Usage

1. **Data Preparation**
   - Place JSON data files in the directory specified in `file_path`.
   - Each JSON should contain fields for raw text and multi-class labels.

2. **Model Training**
   - Run all cells sequentially to train the model on the dataset.
   - Adjust hyperparameters within the training cell if needed.

3. **Evaluation**
   - View accuracy scores and visualizations to assess model performance.

## Requirements

- Install the required libraries:
  ```bash
  pip install matplotlib pandas numpy torch transformers openpyxl
