# Hamas Reports Classifier

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

- Python 3.12
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
 ```

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/arabic-text-classifier.git
    cd arabic-text-classifier
    ```

2. **Download NLTK stopwords**:
    ```sh
    python -c "import nltk; nltk.download('stopwords')"
    ```

3. **Prepare your data**:
   - Organize your text data into files named "level1"..."level4".
   - Run make_categories.py - it will create the folder categories.
   - Each subfolder should represent a category and contain text files for that category.
   - The folder structure should look like this:
     ```
      HamasModel/
      └── model/
          └── data/
              └── categories/
                  ├── category1/
                  │   ├── file1.txt
                  │   └── file2.txt
                  ├── category2/
                  │   ├── file1.txt
                  │   └── file2.txt
                  ├── category3/
                  └── category4/

     ```

## Usage

1. **Run the application**:
    ```sh
    python main.py
    ```

2. **GUI**:
    - Enter the text you want to classify in the text box.
    - Click the "Classify" button to get the predicted category.
    - Use the "Clear" button to clear the input text.
    - Use the "Exit" button to close the application.

## Code Explanation

### main.py
The entry point of the application. It orchestrates loading data, preprocessing text, computing the TF-IDF matrix, training the classifier, and launching the GUI.

### data_loader.py
Contains the `load_text_data` function, which reads text files from the specified directory, assigns labels, and returns shuffled texts and labels.

### preprocessing.py
Contains the `preprocess_text` function, which preprocesses the input text by removing unwanted characters and applying stemming.

### tfidf.py
Contains the `compute_tfidf_matrix` function, which computes the TF-IDF matrix for the preprocessed texts using `TfidfVectorizer`.

### model.py
Contains functions for model training and evaluation:
- `hyperparameter_tuning`: Performs hyperparameter tuning for `RandomForestClassifier` using `GridSearchCV`.
- `train_and_evaluate_classifier`: Splits the data, handles class imbalance with SMOTE, trains the classifier, evaluates it, and plots feature importance and confusion matrix.

### gui.py
Contains functions for creating the GUI using Tkinter:
- `classify_text`: Classifies a given text using the trained classifier.
- `create_gui`: Sets up the GUI with text entry, buttons, and result display.

### utils.py
Contains utility functions:
- `calculate_custom_accuracy`: Calculates custom accuracy considering the distance between categories.

## Acknowledgements

This project uses the following open-source libraries:
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [NLTK](https://www.nltk.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)

