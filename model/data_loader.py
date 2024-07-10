import os
from sklearn.utils import shuffle

def load_text_data(folder_path):
    """
    Load text data from the specified folder path.

    Args:
        folder_path (str): Path to the folder containing text data.

    Returns:
        tuple: Shuffled texts and labels.
    """
    texts = []
    labels = []
    categories = os.listdir(folder_path)
    for idx, category in enumerate(categories):
        files = os.listdir(os.path.join(folder_path, category))
        for file in files:
            try:
                with open(os.path.join(folder_path, category, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
                    labels.append(idx + 1)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
    return shuffle(texts, labels, random_state=42)
