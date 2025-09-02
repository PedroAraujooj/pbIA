import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# 1) Leitura do CSV a partir do link
URL = "https://raw.githubusercontent.com/professortiagoinfnet/inteligencia_artificial/refs/heads/main/sonar_dataset.csv"
df = pd.read_csv(URL, header=None)

# 2) Separação de features e rótulo
X = df.iloc[:, :-1].values
y = LabelEncoder().fit_transform(df.iloc[:, -1])   # 'R'→0, 'M'→1

# 3) Divisão treino / teste (estratificada)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 4) Pipeline: escala → PCA → árvore
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("dt", DecisionTreeClassifier(random_state=42)),
])

# 5) Grade resumida de hiperparâmetros (inclui pruning com ccp_alpha)
param_grid = {
    "pca__n_components": [15, 30, None],   # pode ajustar
    "dt__max_depth": [None, 5],
    "dt__min_samples_split": [2, 5],
    "dt__min_samples_leaf": [1, 2],
    "dt__ccp_alpha": [0.0, 0.001],         # 0 = sem poda
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

# 6) Avaliação no teste
best = grid.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
spec = tn / (tn + fp)

print("Melhores hiperparâmetros:", grid.best_params_)
print("Acurácia:",              round(accuracy_score(y_test, y_pred), 4))
print("Precisão:",              round(precision_score(y_test, y_pred), 4))
print("Recall / Sensibilidade:", round(recall_score(y_test, y_pred), 4))
print("Especificidade:",        round(spec, 4))
print("F1-score:",              round(f1_score(y_test, y_pred), 4))
print("AUC-ROC:",               round(roc_auc_score(y_test, y_prob), 4))
