from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_model():
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    return model

def train_model(model, data, target):
    if not model or not data or not target:
        print("Error: model, data, or target is missing.")
        return None
    try:
        model.fit(data, target)
        return model
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None


def predict(model, data):
    if not model or not data:
        print("Error: model or data is missing.")
        return None
    try:
        predicted = model.predict(data)
        return predicted
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None
