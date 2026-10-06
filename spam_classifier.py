"""Tiny spam classifier using TF-IDF and Naive Bayes (scikit-learn)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from data import MESSAGES


def train(messages=MESSAGES, seed=42):
    texts = [text for text, _ in messages]
    labels = [label for _, label in messages]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, accuracy


def predict(model, text):
    return model.predict([text])[0]


if __name__ == "__main__":
    model, accuracy = train()
    print(f"Test accuracy: {accuracy:.2f}")
    for sample in [
        "Congratulations! You won a free prize, click now",
        "Are we still meeting for the project tomorrow?",
    ]:
        print(f"{predict(model, sample):<5} <- {sample}")
