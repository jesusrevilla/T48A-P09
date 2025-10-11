from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_metrics(y_true, y_pred, average='macro'):
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    return report
