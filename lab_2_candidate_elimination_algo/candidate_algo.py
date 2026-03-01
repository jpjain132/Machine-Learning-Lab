import pandas as pd
import numpy as np

data = pd.read_csv("C:\D_drive_new\semester_6\Machine_learning_lab\lab_2\Enjoy-Sports.csv")

concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def candidate_elimination(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?' for _ in range(len(specific_h))]]

    for i, h in enumerate(concepts):
        if target[i].lower() == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
        else:
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[0][x] = specific_h[x]
                else:
                    general_h[0][x] = '?'

    return specific_h, general_h

s, g = candidate_elimination(concepts, target)
print("Specific Hypothesis:", s)
print("General Hypothesis:", g)