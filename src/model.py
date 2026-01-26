from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_model():
    #Se crea el pipeline que combina el preprocesamiento de texto y el modelo
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    return model

def train_model(model, data, target):
    #Se entrena el pipeline completo
    model.fit(data, target)
    return model

def predict(model, data):
     return model.predict(data)
