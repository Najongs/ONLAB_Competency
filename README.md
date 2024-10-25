# Multi-Class Classification for Self-Introduction Texts

This project focuses on multi-class classification using self-introduction text data. The notebook preprocesses raw data and prepares it for training a machine learning model. The goal is to predict multiple classes from the text by applying a deep learning model.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [GPU Setup](#gpu-setup)
- [Data Preprocessing](#data-preprocessing)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The primary goal of this project is to apply a multi-class classification model to predict various categories from self-introduction texts. The project uses the Huggingface Transformers library and PyTorch to build and train a sequence classification model.

## Features

- **Data Preprocessing**:
  - Loads raw text data from JSON files.
  - Cleans text by removing special characters.
  - Converts multi-label data into a one-hot encoded format for classification.
  
- **GPU Setup**: 
  - Automatically detects available GPUs and configures the environment for training models using CUDA.

- **Model Training**: 
  - Uses PyTorch and Huggingface's transformers for sequence classification tasks.
  - Evaluates model performance using accuracy metrics.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
