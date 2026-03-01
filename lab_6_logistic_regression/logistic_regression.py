import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\D_drive_new\semester_6\Machine_learning_lab\lab_6\lgr.csv")

print(df)

X = df[["hours"]]
y = df["pass"]

model = LogisticRegression()
model.fit(X, y)

X_new = pd.DataFrame({"hours": [5, 10]})

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

print("Prediction for 5 hours:", predictions[0])
print("Probability:", probabilities[0][1])

print("Prediction for 10 hours:", predictions[1])
print("Probability:", probabilities[1][1])

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Actual Data')

x_range = np.linspace(X.min(), X.max(), 100)
y_prob = model.predict_proba(x_range.reshape(-1, 1))[:, 1]

plt.plot(x_range, y_prob, color='black', linewidth=2, label="Logistic Regression Curve")

plt.axhline(0.5, color='black', linestyle='--', alpha=0.5, label="Decision Threshold (0.5)")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()