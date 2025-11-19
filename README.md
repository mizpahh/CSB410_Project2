# Optimizing Deep Learning Pipelines — Sign Language MNIST

This project explores how different optimization techniques affect the performance, stability, and generalization of deep neural networks trained on **Sign Language MNIST**. The goal is to classify 24 grayscale hand-sign images (28×28 pixels) representing letters A–Y (excluding J and Z).  

We experiment with:

- **Optimizers**: Adam, SGD, RMSProp  
- **Regularization techniques**: Dropout, Batch Normalization, L2 Regularization  
- **Network architectures**: Dense models with varying layers and neurons  

---

## 📁 Repository Structure

project/

│
├── notebook.ipynb # Main project notebook
├── README.md # This file
├── requirements.yml # Conda environment dependencies
│
└── data/
├── sign_mnist_train.csv # Training dataset
└── sign_mnist_test.csv # Testing dataset


All dataset files should be placed in the **data/** folder. The notebook uses these files directly.

---

```bash
conda env create -f requirements.yml
conda activate sl-mnist-env
jupyter notebook

Open notebook.ipynb and execute all cells top-to-bottom.

-> What the Notebook Covers

Data Loading & Preprocessing

Normalization of pixel values

Reshaping to (28,28,1)

Label encoding + one-hot conversion

Exploratory Data Analysis

Class distribution

Sample image visualization

Baseline Model

Dense(256) → Dense(128) → Dense(num_classes)

Adam optimizer

5+ epochs training

Optimized Models

Adam, SGD, RMSProp

Dropout, Batch Normalization, L2 Regularization

Larger architecture: Dense(512) → Dense(256) → Dense(num_classes)

Evaluation

Accuracy & loss curves

Confusion matrices

Classification reports

Summary comparison table

Reflection & Ethical Considerations

Analysis of optimizer and regularization impact

Hardest classes to classify

Future improvements

Ethical discussion: accessibility, fairness, and risk


-> Key Findings

Optimized models significantly outperformed the baseline network.

Batch Normalization + Adam produced the most stable and highest-accuracy model.

Hardest classes were visually similar letters (e.g., M/N, U/V, D/F).

Regularization (Dropout/L2) helped prevent overfitting but must be tuned carefully.

Test accuracy exceeding 75% was achievable with Adam + BatchNorm + Dropout + L2.


-> Dependencies

All dependencies are listed in requirements.yml:

Python 3.10

TensorFlow 2.18

Keras

NumPy

Pandas

Matplotlib

Seaborn

scikit-learn

Jupyter


-> Acknowledgements

Kaggle: Sign Language MNIST Dataset

Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn

Course: Deep Learning / Machine Learning (Project 2)