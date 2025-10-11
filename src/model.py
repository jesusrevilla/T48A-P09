from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_model():
    return make_pipeline(TfidfVectorizer(), MultinomialNB())

def train_model(model, data, target):
    return model.fit(data, target)

def predict(model, data):
    predicted = model.predict(data)
    return predicted
