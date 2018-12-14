"""Conduct multi-class classification."""


from MultiClassClassification import multi_class_classification
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv("features with R2.csv")
inputs = data[data.keys()[1:11]]
outputs = data[data.keys()[11:]]
targetnames = ["SVM", "GPR", "NN", "EN"]
targets = outputs.idxmax(axis=1)
x_train, x_test, target_train, target_test, output_train, output_test = train_test_split(
    inputs, targets, outputs
)
x_t1, x_t2, t_t1, t_t2, o_t1, o_t2 = train_test_split(
    x_train, target_train, output_train
)
classifier = multi_class_classification(x_train, output_train)
predictions = classifier.predict(x_test, targetnames)
con_mat = confusion_matrix(target_test, predictions)

print(con_mat)
lent = len(target_test)
lenp = len(predictions)
cost_mat = output_test
cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
cost = classifier.cost_loss(x_test, cost_mat)
# print(lent == lenp, 'Cost based classification', cost/lenp)
cbc = cost / lenp
""" classifier.optmize_weights(x_t2, o_t2)
w = classifier.weights
cost = classifier.weightedcost(w, x_test, cost_mat)
print(lent == lenp, 'Weighted Votes Cost based classification', cost) """
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
lenp = len(predictions)
for index in range(num_samples):
    cost += cost_mat.iloc[index][predictions[index]]
# print('Cost independent classification', cost/lenp)
cic = cost / lenp
classes = targetnames
cost_mat = output_train

votes = pd.DataFrame(
    np.zeros(shape=(len(x_test), len(targetnames))), columns=targetnames
)
for index1 in range(0, num_classes):  # -1 maybe?
    for index2 in range(index1 + 1, num_classes):  # -1 maybe?
        current_classes = [classes[index1], classes[index2]]
        cost = cost_mat[current_classes]
        cost = np.abs(cost.sub(cost.max(axis=1), axis=0))
        target = cost.idxmin(axis=1)
        # targets = np.asarray(cost[cost.columns[0]].astype(bool).astype(int))
        clf = BC()
        clf = clf.fit(x_train, target)
        outputcheck = clf.predict(x_test)
        for id in range(len(outputcheck)):
            votes[outputcheck[id]][id] += 1
predictions = votes.idxmax(axis=1)
con_mat = confusion_matrix(target_test, predictions)
print(con_mat)
cost = 0
cost_mat = -cost_mat.sub(cost_mat.max(axis=1), axis=0)
for index in range(num_samples):
    cost += cost_mat.iloc[index][predictions[index]]
# print('Cost independent voting classification', cost/lenp)
cvc = cost / lenp
print("[", cbc, ",", cic, ",", cvc, "]")

