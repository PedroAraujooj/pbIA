import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    RocCurveDisplay,
    roc_auc_score,
)

# 1) -------------------------------------------------------------------
# Leitura dos CSVs (sem cabeçalho); cada linha: texto , assunto , data
print("Carrega dados")
FAKE_URL = (
    "https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/"
    "refs/heads/main/Fake.csv"
)
TRUE_URL = (
    "https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/"
    "refs/heads/main/True.csv"
)

col_names = ["title", "text", "subject", "date"]
fake_df = pd.read_csv(FAKE_URL, header=None, names=col_names)
true_df = pd.read_csv(TRUE_URL, header=None, names=col_names)

# mantemos apenas o texto e criamos rótulo binário
fake_df = fake_df[["text"]].assign(label=1)  # 1 = Fake
true_df = true_df[["text"]].assign(label=0)  # 0 = True

data = pd.concat([fake_df, true_df], ignore_index=True)

# 2) -------------------------------------------------------------------
# Conversão texto → vetores TF-IDF
print("Conversão texto → vetores TF-IDF")

vectorizer = TfidfVectorizer(
    stop_words="english",  # remove stop-words em inglês
    max_df=0.7,  # ignora termos muito frequentes
    min_df=5,  # ignora termos muito raros
    ngram_range=(1, 2)  # uni- e bi-gramas costumam ajudar em notícias
)
X = vectorizer.fit_transform(data["text"])
y = data["label"].values

# 3) -------------------------------------------------------------------
# Separação treino / teste HOLD-OUT (20 % para teste final)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4) -------------------------------------------------------------------
# Buscamos o melhor k via validação cruzada estratificada
print("Buscamos o melhor k via validação cruzada estratificada")
k_values = range(3, 52, 2)  # apenas ímpares evitam empates
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_means = []
cv_stds = []

for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())

# --- visualiza impacto de k na acurácia -------------------------------
plt.figure(figsize=(8, 4))
plt.errorbar(k_values, cv_means, yerr=cv_stds, marker="o", capsize=3)
plt.title("Validação cruzada ‒ Acurácia × k (K-NN)")
plt.xlabel("k (vizinhos)")
plt.ylabel("Acurácia média (±1 desvio-padrão)")
plt.grid(True)
plt.show()

best_k = k_values[int(np.argmax(cv_means))]
print(f"Melhor k (média CV): {best_k} | Acurácia={max(cv_means):.4f}")

# 5) -------------------------------------------------------------------
# Treino final com o melhor k
print("Treino final com o melhor k")
best_clf = KNeighborsClassifier(n_neighbors=best_k, weights="distance", metric="cosine")
best_clf.fit(X_train, y_train)

# 6) -------------------------------------------------------------------
# Avaliação no conjunto de teste hold-out
y_pred = best_clf.predict(X_test)
y_proba = best_clf.predict_proba(X_test)[:, 1]  # prob. classe positiva (Fake)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)  # sensibilidade (TPR)
f1 = f1_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf.ravel()
spec = tn / (tn + fp)  # especificidade
auc = roc_auc_score(y_test, y_proba)

print("\nMétricas no TESTE")
print(f"Acurácia     : {acc:.4f}")
print(f"Precisão     : {prec:.4f}")
print(f"Recall (sens): {rec:.4f}")
print(f"Especificidade: {spec:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"AUC-ROC      : {auc:.4f}")
print("\nMatriz de confusão:")
print(conf)

# 7) -------------------------------------------------------------------
# Curva ROC
RocCurveDisplay.from_predictions(y_test, y_proba, name="KNN ( TF-IDF )")
plt.show()
