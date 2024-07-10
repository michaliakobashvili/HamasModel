# HamasModel
Analysis of Hamas Reports: Machine Learning Perspective
# Arabic Text Classifier

This project is an Arabic text classification application that uses a RandomForest classifier with TF-IDF feature extraction. The application includes a graphical user interface (GUI) built with Tkinter for easy interaction.

## Features

- **Data Loading**: Load and preprocess text data from a specified folder.
- **Text Preprocessing**: Remove digits, English words, punctuation, and stopwords, and apply stemming to the text.
- **TF-IDF Computation**: Compute the TF-IDF matrix for the preprocessed texts.
- **Model Training**: Train and evaluate a RandomForest classifier using the TF-IDF matrix.
- **Hyperparameter Tuning**: Perform hyperparameter tuning using GridSearchCV.
- **Custom Accuracy Calculation**: Calculate a custom accuracy score considering the distance between categories.
- **GUI**: Classify input text through a user-friendly graphical interface.

## Project Structure

- `main.py`: Entry point of the application.
- `data_loader.py`: Functions related to loading data.
- `preprocessing.py`: Text preprocessing functions.
- `tfidf.py`: Functions for computing the TF-IDF matrix.
- `model.py`: Functions for training and evaluating the model.
- `gui.py`: GUI-related functions.
- `utils.py`: Utility functions.

## Prerequisites

- Python 3.x
- The following Python libraries:
  - `scikit-learn`
  - `imblearn`
  - `nltk`
  - `matplotlib`
  - `seaborn`
  - `tkinter`
  
You can install the required libraries using `pip`:
```sh
pip install scikit-learn imbalanced-learn nltk matplotlib seaborn
