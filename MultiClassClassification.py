"""Multi class cost-based classification.

This function conducts cost-based multi class classification by creating
cost-based pair-wise classifiers which vote to determine the predicted class.
Pair-wise classifiers are created using the package Costcla.
"""


import pandas as pd
import numpy as np
from costcla.metrics import cost_loss
from costcla.models import CostSensitiveRandomForestClassifier as CSRF


class multi_class_classifier():
    """An ensemble of cost-based binary classifiers."""

    def __init__(self, num_classes):
        """Create empty structure for classifier."""
        self.classifier = []
        self.cost = 0
        self.num_classifiers = (num_classes * (num_classes - 1)) / 2
        self.classes = []
        return

    def predict(self, x, allclasses):
        """Predict on x."""
        votes = pd.DataFrame(np.zeros(shape=(len(x), len(allclasses))),
                             columns=allclasses)
        for index in range(self.num_classifiers+1):
            classifier = self.classifier[index]
            classes = self.classes[index]
            y = classifier.predict(X)
            for id in len(y):
                y_current = classes[y[id]]
                votes[y_current][id] += 1
        predictions = votes.idxmax(axis=1)
        return(predictions)

    def cost_loss(self, x, y, cost_mat):
        """Calculate Cost of a given classifier."""
        predictions = self.predict(x, list(cost_mat.columns))
        num_samples, num_classes = y.shape
        for index in range(num_samples):
            self.cost += cost_mat.iloc[index][predictions[index]]
        return


def multi_class_classification(x_train, cost_mat):
    """Conduct pair-wise cost based classification."""
    classes = list(cost_mat.columns)
    num_classes = len(classes)

    classifier = multi_class_classifier(num_classes)
    classifier_index = 0
    for index1 in range(0, num_classes):  # -1 maybe?
        for index2 in range(index1, num_classes):  # -1 maybe?
            if classifier_index > classifier.num_classifiers:
                print("Number of Classifiers Error")
                break
            current_classes = [classes[index1], classes[index2]]
            cost = cost_mat[:, current_classes]
            cost = -cost.sub(cost.max(axis=1), axis=0)
            targets = cost[cost.columns[0]].astype(bool).astype(int)
            cost = np.hstack((np.asarray(cost), np.zeros(cost.shape)))
            classifier.classifier[classifier_index] = CSRF()
            classifier.classifier[classifier_index].fit(x_train, targets, cost)
            classifier.classes.append(current_classes)
            classifier_index += 1
    classifier.cost_loss(x_train, cost_mat)
    return(classifier)