from sklearn.naive_bayes import MultinomialNB
 
def create_model():
    pass
    """
    Crea el modelo con un pipeline que usa TfidfVectorizer y MultinomialNB.
    """
    return make_pipeline(TfidfVectorizer(), MultinomialNB())
 
def train_model(model, data, target):
    pass
    """
    Entrena el modelo con los datos de entrada.
    """
    model.fit(data, target)
    return model
 
def predict(model, data):
    pass
    """
    Realiza predicciones con el modelo entrenado.
    """
    return model.predict(data)
