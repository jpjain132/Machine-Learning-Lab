import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\D_drive_new\semester_6\Machine_learning_lab\lab_1\1.csv")

concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def find_s(concepts, target):
    for i, val in enumerate(target):
        if val.lower() == "yes":
            hypothesis = concepts[i].copy()
            break

    for i, val in enumerate(concepts):
        if target[i].lower() == "yes":
            for j in range(len(hypothesis)):
                if val[j] != hypothesis[j]:
                    hypothesis[j] = '?'
    return hypothesis

print("Final Hypothesis:", find_s(concepts, target))