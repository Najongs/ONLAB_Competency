Multi-Class Classification for Self-Introduction Texts
This project focuses on multi-class classification using self-introduction text data. The notebook preprocesses raw data and prepares it for training a machine learning model. The goal is to predict multiple classes from the text by applying a deep learning model.

Table of Contents
Project Overview
Features
Installation
Usage
GPU Setup
Data Preprocessing
Requirements
Contributing
License
Project Overview
The primary goal of this project is to apply a multi-class classification model to predict various categories from self-introduction texts. The project uses the Huggingface Transformers library and PyTorch to build and train a sequence classification model.

Features
Data Preprocessing:

Loads raw text data from JSON files.
Cleans text by removing special characters.
Converts multi-label data into a one-hot encoded format for classification.
GPU Setup:

Automatically detects available GPUs and configures the environment for training models using CUDA.
Model Training:

Uses PyTorch and Huggingface's transformers for sequence classification tasks.
Evaluates model performance using accuracy metrics.
Installation
Clone the repository:

bash
코드 복사
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install the required dependencies:

bash
코드 복사
pip install -r requirements.txt
Alternatively, manually install the necessary libraries:

bash
코드 복사
pip install matplotlib pandas numpy torch transformers scikit-learn tqdm openpyxl
Usage
Ensure that your raw JSON data is stored in the appropriate directory, as specified in the notebook.

Run the Jupyter notebook to process the data and train the model:

bash
코드 복사
jupyter notebook Multi_Class.ipynb
The notebook will:

Load the data,
Preprocess it,
Train the model, and
Evaluate the performance.
GPU Setup
The notebook automatically detects available GPUs and sets up the environment to utilize them.
For custom GPU settings, modify the following lines in the notebook:
python
코드 복사
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2"  # Set the GPUs you want to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Data Preprocessing
The notebook reads JSON files containing self-introduction texts and their corresponding classes. It processes the text by:

Removing special characters,
Tokenizing the text,
Converting the class labels into one-hot encoded format.
This processed data is then ready to be fed into a model for training.

Requirements
Python 3.8+
PyTorch
Huggingface Transformers
Scikit-learn
GPU (optional, but recommended for faster training)
Contributing
Contributions are welcome! If you'd like to improve the model or add new features, feel free to fork this repository and open a pull request.

Fork the repository
Create a new branch (git checkout -b feature-branch)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature-branch)
Open a Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.
