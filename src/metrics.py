from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_metrics(y_true, y_pred, average='macro'):
    print(metrics.classification_report(test.target, predicted, target_names=test.target_names))
