import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

def get_classification_scores(clt, x_test, y_test):
    # clt: classifier
    # test: test set

    # get the predictions
    y_pred = clt.predict(x_test)

    # get the true labels
    y_true = y_test

    # get true positives, false positives, true negatives, and false negatives
    tp = np.sum(y_pred * y_true)
    fp = np.sum(y_pred * (1 - y_true))
    fn = np.sum((1 - y_pred) * y_true)
    tn = np.sum((1 - y_pred) * (1 - y_true))

    # Recall
    r = tp / (tp + fn)

    # Precision
    p = tp / (tp + fp)

    # F1 score
    f1 = 2 * p * r / (p + r)

    # Accuracy
    a = (tp + tn) / (tp + tn + fp + fn)

    # AUC score
    auc = metrics.roc_auc_score(y_true, y_pred)

    #Specificity
    s = tn / (tn + fp)

    #gmean
    g = np.sqrt(r * s)

    # return the scores
    return r, p, f1, a, s, g, auc


def scores_auc(clt, test):
    # clt: classifier
    # test: test set

    # get the predictions
    y_pred = clt.predict(test)

    # get the true labels
    y_true = test[:, -1]

    # roc_curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    # plot the roc curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

# clt = None
# test = None

# recall, precision, f1, accuracy, auc = get_classification_scores(clt, test)
