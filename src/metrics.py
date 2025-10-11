from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_metrics(y_true, y_pred, average='macro'):
    precision = precision_score(y_true, y_pred, average=average, zero_division=0) #y_true son las etiquetas verdaderas, y_pred las predichas y average es el tipo de promedio a usar que es en este caso macro 
                    Por defecto usa 'macro'.
    recall = recall_score(y_true, y_pred, average=average, zero_division=0) #Sensibilidad o Exhaustividad
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
