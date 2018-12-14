"""Multi class cost-based classification.

This function conducts cost-based multi class classification by creating
cost-based pair-wise classifiers which vote to determine the predicted class.
Pair-wise classifiers are created using the package Costcla.
"""


import pandas as pd
import numpy as np
from costcla.metrics import cost_loss
from costcla.models import CostSensitiveBaggingClassifier as CSBC
from scipy.optimize import minimize


class multi_class_classifier:
    """An ensemble of cost-based binary classifiers."""

    def __init__(self, num_classes):
        """Create empty structure for classifier."""
        self.classifier = []
        self.cost = 0
        self.num_classifiers = int((num_classes * (num_classes - 1)) / 2)
        self.classes = []
        self.weights = np.array([1 / self.num_classifiers] * self.num_classifiers)
        return

    def predict(self, x, allclasses):
        """Predict on x."""
        votes = pd.DataFrame(
            np.zeros(shape=(len(x), len(allclasses))), columns=allclasses
        )
        for index in range(self.num_classifiers):
            classifier = self.classifier[index]
            classes = self.classes[index]
            y = classifier.predict(x.values)
            for id in range(len(y)):
                votes[y[id]][id] += 1  # SHADY
        predictions = votes.idxmax(axis=1)
        return predictions

    def cost_loss(self, x, cost_mat):
        """Calculate Cost of a given classifier."""
        self.cost = 0
        predictions = self.predict(x, list(cost_mat.columns))
        num_samples, num_classes = cost_mat.shape
        for index in range(num_samples):
            self.cost += cost_mat.iloc[index][predictions[index]]
        return self.cost

    def optmize_weights(self, x_train, cost_mat):
        """Testing weighted voting."""
        cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)

        def con(weight):
            return sum(weight) - 1

        cons = {"type": "eq", "fun": con}
        weights = self.weights
        bounds = [[0, 5]] * self.num_classifiers
        optres = minimize(
            self.weightedcost,
            x0=weights,
            args=(x_train, cost_mat),
            bounds=bounds,
            constraints=cons,
        )
        self.weights = optres.x
        self.cost = optres.fun

    def weightedcost(self, w, x_train, cost_mat):
        """Calculate Cost of a given classifier."""
        predictions = self.weightedpredict(w, x_train, list(cost_mat.columns))
        num_samples, num_classes = cost_mat.shape
        cost = 0
        for index in range(num_samples):
            cost += cost_mat.iloc[index][predictions[index]]
        return cost / num_samples

    def weightedpredict(self, w, x, allclasses):
        """Predict on x based on weights."""
        if w is None:
            w = self.weights
        votes = pd.DataFrame(
            np.zeros(shape=(len(x), len(allclasses))), columns=allclasses
        )
        for index in range(self.num_classifiers):
            classifier = self.classifier[index]
            classes = self.classes[index]
            weight = w[index]
            y = classifier.predict(x.values)
            for id in range(len(y)):
                votes[y[id]][id] += weight  # SHADY
        predictions = votes.idxmax(axis=1)
        return predictions


def multi_class_classification(x_train, cost_mat):
    """Conduct pair-wise cost based classification."""
    classes = list(cost_mat.columns)
    num_classes = len(classes)

    classifier = multi_class_classifier(num_classes)
    classifier_index = 0
    for index1 in range(0, num_classes):  # -1 maybe?
        for index2 in range(index1 + 1, num_classes):  # -1 maybe?
            print("Fitting classifier ", classifier_index + 1)
            if classifier_index > classifier.num_classifiers:
                print("Number of Classifiers Error")
                break
            current_classes = [classes[index1], classes[index2]]
            cost = cost_mat[current_classes]
            cost = np.abs(cost.sub(cost.max(axis=1), axis=0))
            targets = cost.idxmin(axis=1)
            # targets = np.asarray(cost[cost.columns[0]].astype(bool).astype(int))
            cost = cost[cost.columns[::-1]]
            cost = np.hstack((np.asarray(cost), np.zeros(cost.shape)))
            classifier.classifier.append(CSBC())
            classifier.classifier[classifier_index].fit(
                x_train.values, targets.values, cost
            )
            classifier.classes.append(current_classes)
            classifier_index += 1
    cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
    classifier.cost_loss(x_train, cost_mat)
    return classifier

