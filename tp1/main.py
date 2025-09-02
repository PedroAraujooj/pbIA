import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv(
    "https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/refs/heads/main/heart.csv"
)
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_val_num = scaler.transform(X_val[num_cols])

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[cat_cols])
X_val_cat = ohe.transform(X_val[cat_cols])

X_train_proc = np.hstack([X_train_num, X_train_cat])
X_val_proc = np.hstack([X_val_num, X_val_cat])

k_values = range(1, 16, 2)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_proc, y_train)
    y_pred = knn.predict(X_val_proc)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
    print(f"k = {k:<2} | Validation accuracy = {acc:.3f}")

plt.plot(list(k_values), accuracies, marker="o")
plt.title("KNN – acurácia de validação vs. k")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia")
plt.grid(True)
plt.show()
