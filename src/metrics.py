from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_metrics(y_true, y_pred, average='macro'):
    precision = presicion_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=avearge)
    f1 = f1_score(y_true, y_pred, average=average)

return {
    'presicion' : precision,
    'recall': recall,
    'f1_score': f1
}

