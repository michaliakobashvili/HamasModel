import tkinter as tk
from tkinter import messagebox
from preprocessing import preprocess_text

def classify_text(text, classifier, tfidf_vectorizer):
    """
    Classify a given text using the trained classifier.

    Args:
        text (str): Text to classify.
        classifier (RandomForestClassifier): Trained classifier.
        tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.

    Returns:
        int: Predicted category of the text.
    """
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prediction = classifier.predict(text_tfidf)
    return prediction[0]

def create_gui(classifier, tfidf_vectorizer):
    """
    Create the GUI for text classification.

    Args:
        classifier (RandomForestClassifier): Trained classifier.
        tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """

    def classify_and_display():
        text = text_entry.get("1.0", tk.END).strip()
        if text:
            category = classify_text(text, classifier, tfidf_vectorizer)
            result_label.config(text=f"Predicted Category: {category}")
        else:
            messagebox.showwarning("Input Error", "Please enter some text to classify.")

    def clear_input():
        text_entry.delete("1.0", tk.END)
        result_label.config(text="")

    def exit_program():
        root.destroy()

    root = tk.Tk()
    root.title("Arabic Text Classifier")
    root.configure(bg="#87CEEB")  # Set background color

    frame = tk.Frame(root, padx=20, pady=20, bg="#87CEEB")
    frame.pack(padx=20, pady=20)

    tk.Label(frame, text="Enter the text to classify:", bg="#87CEEB", font=("Helvetica", 12)).pack()

    text_entry = tk.Text(frame, wrap=tk.WORD, width=50, height=10, font=("Helvetica", 12))
    text_entry.pack(pady=10)

    classify_button = tk.Button(frame, text="Classify", command=classify_and_display, bg="#4CAF50", fg="white", font=("Helvetica", 12), padx=10, pady=5)
    classify_button.pack(pady=5)

    result_label = tk.Label(frame, text="", font=("Helvetica", 14), bg="#87CEEB")
    result_label.pack(pady=10)

    button_frame = tk.Frame(frame, bg="#87CEEB")
    button_frame.pack(pady=10)

    clear_button = tk.Button(button_frame, text="Clear", command=clear_input, bg="#FF9800", fg="white", font=("Helvetica", 12), padx=10, pady=5)
    clear_button.pack(side=tk.LEFT, padx=5)

    exit_button = tk.Button(button_frame, text="Exit", command=exit_program, bg="#F44336", fg="white", font=("Helvetica", 12), padx=10, pady=5)
    exit_button.pack(side=tk.RIGHT, padx=5)

    root.mainloop()
