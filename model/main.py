import os
from data_loader import load_text_data
from preprocessing import preprocess_text
from tfidf import compute_tfidf_matrix
from model import train_and_evaluate_classifier
from gui import create_gui

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, './data/categories')

    if not os.path.exists(folder_path):
        folder_path = os.path.join(os.path.dirname(script_dir), 'categories')

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The categories folder was not found in {script_dir} or its parent directory.")

    texts, labels = load_text_data(folder_path)
    preprocessed_texts = [preprocess_text(text) for text in texts]
    tfidf_matrix, tfidf_vectorizer = compute_tfidf_matrix(preprocessed_texts)

    print("RandomForest Classifier Results:")
    classifier, tfidf_vectorizer = train_and_evaluate_classifier(tfidf_matrix, labels, tfidf_vectorizer)

    create_gui(classifier, tfidf_vectorizer)

if __name__ == "__main__":
    main()
