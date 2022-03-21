from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

def Vis(clf,x_test,y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred) 
    f1score = metrics.f1_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)

    data_log = {'accuracy': accuracy,
                'f1score': f1score,
                'recall': recall,
                'precision': precision,
                }
    print("Classification report:\n", classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test)  
    print(data_log)
    return "Done..."