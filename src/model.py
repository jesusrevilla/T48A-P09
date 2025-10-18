from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_model():
    """
    Crea el modelo con un pipeline que usa TfidfVectorizer y MultinomialNB.
    """
    return make_pipeline(TfidfVectorizer(), MultinomialNB())

def train_model(model, data, target):
    """
    Entrena el modelo con los datos de entrada.
    """
    model.fit(data, target)
    return model

def predict(model, data):
    """
    Realiza predicciones con el modelo entrenado.
    """
