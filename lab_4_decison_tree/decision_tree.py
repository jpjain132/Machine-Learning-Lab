import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\D_drive_new\semester_6\Machine_learning_lab\lab_4\Play_Tennis.csv")


data = data.drop(columns=["Day"])

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(x)
print(y)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = 0
    for i in range(len(elements)):
        probability = counts[i] / np.sum(counts)
        entropy_value += -probability * math.log2(probability)
    return entropy_value

def information_gain(data, split_attribute, target_name):
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[split_attribute], return_counts=True)

    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[split_attribute] == values[i]]
        subset_entropy = entropy(subset[target_name])
        weighted_entropy += (counts[i] / np.sum(counts)) * subset_entropy

    IG = total_entropy - weighted_entropy
    return IG

print("Entropy:", entropy(data["Play"]))

print("Information Gain of each attribute:\n")
for feature in x.columns:
    print(feature, ":", information_gain(data, feature, "Play"))

le = LabelEncoder()
x_encoded = x.copy()

for col in x.columns:
    x_encoded[col] = le.fit_transform(x[col])

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_encoded, y_encoded)

plt.figure(figsize=(10, 8))
plot_tree(dt, feature_names=x.columns, class_names=le_target.classes_, filled=True)
plt.title("Decision Tree (ID3)")
plt.show()
