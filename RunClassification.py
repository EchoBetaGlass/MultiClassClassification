"""Conduct multi-class classification."""


from MultiClassClassification import multi_class_classification
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

num_samples = 300
num_variables = 20
inputs = np.random.random((num_samples, num_variables))

sums = pd.DataFrame(np.sum(inputs, axis=1))

cycles = np.pi
theta = np.pi / 5

y1 = np.power(np.sin(cycles * sums + theta), 2)
y2 = np.power(np.sin(cycles * sums + 2 * theta), 2)
y3 = np.power(np.sin(cycles * sums + 3 * theta), 2)
y4 = np.power(np.sin(cycles * sums + 4 * theta), 2)
y5 = np.power(np.sin(cycles * sums + 5 * theta), 2)
y = np.hstack([y1, y2, y3, y4, y5])
targetnames = ["y1", "y2", "y3", "y4", "y5"]
outputs = pd.DataFrame(y, columns=targetnames)
targets = outputs.idxmax(axis=1)
x_train, x_test, target_train, target_test, output_train, output_test = train_test_split(
    inputs, targets, outputs
)
x_t1, x_t2, t_t1, t_t2, o_t1, o_t2 = train_test_split(
    x_train, target_train, output_train
)
classifier = multi_class_classification(x_t1, o_t1)
predictions = classifier.predict(x_test, targetnames)
con_mat = confusion_matrix(target_test, predictions)

print(con_mat)
lent = len(target_test)
lenp = len(predictions)
cost_mat = output_test
cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
cost = classifier.cost_loss(x_test, cost_mat)
print(lent == lenp, "Cost based classification", cost / lenp)
classifier.optmize_weights(x_t2, o_t2)
w = classifier.weights
cost = classifier.weightedcost(w, x_test, cost_mat)
print(lent == lenp, "Weighted Votes Cost based classification", cost)
################################################
num_classes = len(targetnames)

clf = BC()
clf = clf.fit(x_train, target_train)
predictions = clf.predict(x_test)
con_mat = confusion_matrix(target_test, predictions)
cost_mat = output_test
cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
print(con_mat)
num_samples, num_classes = cost_mat.shape
cost = 0
for index in range(num_samples):
    cost += cost_mat.iloc[index][predictions[index]]
print("Cost independent classification", cost / lenp)
"""
classes = targetnames
cost_mat = output_train
votes = pd.DataFrame(np.zeros(shape=(len(x_test), len(targetnames))),
                             columns=targetnames)
for index1 in range(0, num_classes):  # -1 maybe?
    for index2 in range(index1+1, num_classes):  # -1 maybe?
        current_classes = [classes[index1], classes[index2]]
        cost = cost_mat[current_classes]
        cost = np.abs(cost.sub(cost.max(axis=1), axis=0))
        target = cost.idxmin(axis=1)
        #targets = np.asarray(cost[cost.columns[0]].astype(bool).astype(int))
        clf = BC()
        clf = clf.fit(x_train, target)
        outputcheck = clf.predict(x_test)
        for id in range(len(outputcheck)):
            votes[outputcheck[id]][id] += 1
predictions = votes.idxmax(axis=1)
con_mat = confusion_matrix(target_test, predictions)
print(con_mat) """
