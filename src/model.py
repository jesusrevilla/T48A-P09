from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_model():
    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    return model

def train_model(model, data, target):
    model.fit(data, target)
    return model

def predict(model, data):
    return model.predict(data)
