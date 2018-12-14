import pandas as pd
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# DTLZ STUFFS
data = pd.read_csv("features with R2.csv")
inputs_train = data[data.keys()[1:11]]
outputs_train = data[data.keys()[11:]]
targetnames = ["SVM", "GPR", "NN", "EN"]
target_train = outputs_train.idxmax(axis=1)

num_classes = len(targetnames)

clf = BC()
clf = clf.fit(inputs_train, target_train)

# WFG STUFFS
data_wfg = pd.read_csv("WFG features with R2.csv")
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
