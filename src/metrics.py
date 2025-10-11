from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_metrics(y_true, y_pred, average='macro'):
    # 1. Precisión
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    
    # 2. Recall
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    # 3. F1-score
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Devolver como tupla en el orden solicitado
    return (precision, recall, f1)
