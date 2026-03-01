import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\D_drive_new\semester_6\Machine_learning_lab\lab_5\naive.csv")

print("Dataset:\n", df)

X = df.drop("Species", axis=1)
y = df["Species"]

encoder = LabelEncoder()
X_encoded = X.apply(encoder.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42)

model = CategoricalNB(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")