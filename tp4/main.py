from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, silhouette_score)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    if df.columns[0].lower().startswith("unnamed") or df.columns[0] == "":
        df = pd.read_csv(csv_path, encoding="utf-8", engine="python", index_col=0)
    return df

def pick_target(df: pd.DataFrame) -> str:
    for c in ["track_genre", "genre", "target", "label", "class", "y"]:
        if c in df.columns:
            return c
    if "explicit" in df.columns:
        return "explicit"
    raise ValueError("Não encontrei coluna alvo (ex.: 'track_genre' ou 'explicit').")

def prepare_xy(df: pd.DataFrame, target_col: str):
    df = df.dropna(subset=[target_col]).copy()
    # bool -> int
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    X = df[num_cols].dropna(axis=0)
    y = df.loc[X.index, target_col]

    le = LabelEncoder()
    y = le.fit_transform(y)
    return X.values, y, num_cols, le

# KMeans com cotovelo e silhueta
def k_diagnostics(X_scaled: np.ndarray, k_min=2, k_max=10):
    k_max = max(k_min + 1, min(k_max, len(X_scaled) - 1))
    ks, inertias, silhouettes = [], [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        ks.append(k)
        inertias.append(km.inertia_)
        labs = km.labels_
        if len(set(labs)) > 1:
            silhouettes.append(silhouette_score(X_scaled, labs))
        else:
            silhouettes.append(np.nan)
    # melhor k pela silhueta (se disponível)
    valid = [(k, s) for k, s in zip(ks, silhouettes) if not np.isnan(s)]
    best_k = max(valid, key=lambda t: t[1])[0] if valid else 3
    return ks, inertias, silhouettes, best_k

def plot_elbow_silhouette(ks, inertias, silhouettes, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inércia")
    plt.title("Método do cotovelo (K-Means)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "cotovelo_kmeans.png", dpi=130)
    plt.close()

    plt.figure()
    plt.plot(ks, silhouettes, marker="o")
    plt.xlabel("k")
    plt.ylabel("Índice de Silhueta")
    plt.title("Silhueta (K-Means)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "silhueta_kmeans.png", dpi=130)
    plt.close()

# Feature de cluster
def cluster_distance_feature(Xtr_s, Xte_s, k: int):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xtr_s)
    dtr = km.transform(Xtr_s).min(axis=1).reshape(-1, 1)
    dte = km.transform(Xte_s).min(axis=1).reshape(-1, 1)
    return dtr, dte

# Modelos e avaliação
def svm_grid():
    model = SVC(probability=True, random_state=RANDOM_STATE)
    grid = [
        {"kernel": ["linear"], "C": [0.1, 1, 10]},
        {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        {"kernel": ["poly"], "C": [0.1, 1], "degree": [2, 3], "gamma": ["scale"]}
    ]
    return model, grid

def rf_grid():
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = {"n_estimators": [200], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3]}
    return model, grid

def fit_best(model, grid, X, y, scoring="f1_macro"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True, verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def metrics(model, Xte, yte):
    yp = model.predict(Xte)
    acc = accuracy_score(yte, yp)
    prec = precision_score(yte, yp, average="macro", zero_division=0)
    rec = recall_score(yte, yp, average="macro", zero_division=0)
    f1 = f1_score(yte, yp, average="macro", zero_division=0)
    auc = np.nan
    try:
        if hasattr(model, "predict_proba"):
            yscore = model.predict_proba(Xte)
        elif hasattr(model, "decision_function"):
            yscore = model.decision_function(Xte)
        else:
            yscore = None
        if yscore is not None:
            lb = LabelBinarizer()
            yb = lb.fit_transform(yte)
            if yb.shape[1] == 1:
                auc = roc_auc_score(yte, yscore[:, 1] if yscore.ndim > 1 else yscore)
            else:
                auc = roc_auc_score(yb, yscore, average="macro", multi_class="ovr")
    except Exception:
        pass
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1, "roc_auc_macro": auc}

def bar_compare(metrics_table: dict, outdir: Path, title_prefix: str):
    labels = list(metrics_table.keys())
    f1_vals = [metrics_table[k]["f1_macro"] for k in labels]
    auc_vals = [0.0 if np.isnan(metrics_table[k]["roc_auc_macro"]) else metrics_table[k]["roc_auc_macro"] for k in labels]

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, f1_vals)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("F1 (macro)")
    plt.title(f"{title_prefix}: F1 (macro)")
    plt.tight_layout()
    plt.savefig(outdir / f"{title_prefix.lower().replace(' ', '_')}_f1.png", dpi=130)
    plt.close()

    plt.figure()
    plt.bar(x, auc_vals)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("ROC-AUC (macro)")
    plt.title(f"{title_prefix}: ROC-AUC (macro)")
    plt.tight_layout()
    plt.savefig(outdir / f"{title_prefix.lower().replace(' ', '_')}_auc.png", dpi=130)
    plt.close()

# Influência de k (SVM)
def scan_k_effect(k_values, Xtr_s, Xte_s, ytr, yte):
    recs = []
    base_svm, base_grid = svm_grid()
    for k in k_values:
        if k < 2 or k >= len(Xtr_s):
            continue
        dtr, dte = cluster_distance_feature(Xtr_s, Xte_s, k)
        Xtr_plus = np.hstack([Xtr_s, dtr])
        Xte_plus = np.hstack([Xte_s, dte])
        best, params, cvscore = fit_best(base_svm, base_grid, Xtr_plus, ytr, scoring="f1_macro")
        mets = metrics(best, Xte_plus, yte)
        recs.append({"k": k, "f1_macro": mets["f1_macro"], "roc_auc_macro": mets["roc_auc_macro"]})
    return pd.DataFrame(recs).sort_values("k")

def plot_k_effect(dfk: pd.DataFrame, outdir: Path):
    if dfk.empty:
        return
    plt.figure()
    plt.plot(dfk["k"], dfk["f1_macro"], marker="o")
    plt.xlabel("k")
    plt.ylabel("F1 (macro) - SVM com feature de cluster")
    plt.title("Influência do número de clusters (k) no F1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "influencia_k_f1.png", dpi=130)
    plt.close()

    if "roc_auc_macro" in dfk.columns and not dfk["roc_auc_macro"].isna().all():
        plt.figure()
        plt.plot(dfk["k"], dfk["roc_auc_macro"], marker="o")
        plt.xlabel("k")
        plt.ylabel("ROC-AUC (macro)")
        plt.title("Influência do número de clusters (k) na AUC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "influencia_k_auc.png", dpi=130)
        plt.close()

# ---------------------------
# Diretrizes (com base nos resultados)
# ---------------------------
def print_guidelines(best_k: int, comp_metrics: dict, dfk: pd.DataFrame):
    delta_f1_svm = comp_metrics["SVM (+cluster)"]["f1_macro"] - comp_metrics["SVM (base)"]["f1_macro"]
    delta_f1_rf  = comp_metrics["RF  (+cluster)"]["f1_macro"]  - comp_metrics["RF  (base)"]["f1_macro"]
    delta_med = np.nanmean([delta_f1_svm, delta_f1_rf])

    print("\n=== Diretrizes práticas ===")
    print(f"- Escolha inicial: k={best_k} (pela silhueta); avalie k±1..2 (vide curvas salvas).")
    print(f"- A feature de cluster (distância ao centróide) {'tende a ajudar' if delta_med>0 else 'pode não ajudar'} (ΔF1 médio≈{delta_med:.4f}).")
    print("- Sempre padronize antes de K-Means e SVM (StandardScaler).")
    print("- SVM: teste kernels linear e RBF primeiro; ajuste C (e gamma no RBF).")
    print("- Random Forest: ajuste max_depth e min_samples_leaf para controlar overfitting.")
    print("- Se o ganho for pequeno, tente: (i) adicionar mais de uma distância (p.ex. top-2), "
          "(ii) usar one-hot dos clusters (assignment), (iii) ampliar a faixa de k.")

# MAIN
def main():
    base = Path(__file__).parent
    data_path = base / "dataset.csv"

    outdir = base / "figs2"

    print(f"Lendo: {data_path}")
    df = load_dataset(data_path)
    target = pick_target(df)
    X, y, feat_names, le = prepare_xy(df, target)
    print(f"Alvo: {target} | Amostras: {len(X)} | Features numéricas: {X.shape[1]}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # KMeans diagnostics: cotovelo + silhueta
    ks, inertias, silhouettes, best_k = k_diagnostics(Xtr_s, k_min=2, k_max=min(12, max(3, int(np.sqrt(len(Xtr_s))))))
    plot_elbow_silhouette(ks, inertias, silhouettes, outdir)
    with open(outdir / "k_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({"ks": ks, "inertias": inertias, "silhouettes": silhouettes, "best_k_silhouette": int(best_k)}, f, ensure_ascii=False, indent=2)
    print(f"k (silhueta): {best_k}")

    # Feature de cluster
    dtr, dte = cluster_distance_feature(Xtr_s, Xte_s, k=best_k)
    Xtr_plus = np.hstack([Xtr_s, dtr])
    Xte_plus = np.hstack([Xte_s, dte])

    # ----- SVM -----
    svm0, grid0 = svm_grid()
    svm_base, p_base, cv_base = fit_best(svm0, grid0, Xtr_s,   ytr, scoring="f1_macro")
    svm_plus, p_plus, cv_plus = fit_best(svm0, grid0, Xtr_plus, ytr, scoring="f1_macro")
    m_svm_base = metrics(svm_base, Xte_s, yte)
    m_svm_plus = metrics(svm_plus, Xte_plus, yte)

    # ----- RF -----
    rf0, rgrid = rf_grid()
    rf_base, rp_base, rcv_base = fit_best(rf0, rgrid, Xtr_s,   ytr, scoring="f1_macro")
    rf_plus, rp_plus, rcv_plus = fit_best(rf0, rgrid, Xtr_plus, ytr, scoring="f1_macro")
    m_rf_base  = metrics(rf_base, Xte_s, yte)
    m_rf_plus  = metrics(rf_plus, Xte_plus, yte)

    comp = {
        "SVM (base)": m_svm_base,
        "SVM (+cluster)": m_svm_plus,
        "RF  (base)": m_rf_base,
        "RF  (+cluster)": m_rf_plus,
    }
    print("\n=== Comparação (Teste) ===")
    print(pd.DataFrame(comp).T.round(4))

    # salvar params selecionados
    with open(outdir / "melhores_parametros.json", "w", encoding="utf-8") as f:
        json.dump({
            "SVM_base": {"params": p_base, "cv_f1_macro": cv_base},
            "SVM_plus": {"params": p_plus, "cv_f1_macro": cv_plus},
            "RF_base":  {"params": rp_base, "cv_f1_macro": rcv_base},
            "RF_plus":  {"params": rp_plus, "cv_f1_macro": rcv_plus},
        }, f, ensure_ascii=False, indent=2)

    # gráficos de comparação
    bar_compare(comp, outdir, "Comparação sem vs com feature de cluster")

    # Influência de k no SVM (com feature)
    dfk = scan_k_effect(ks, Xtr_s, Xte_s, ytr, yte)
    dfk.to_csv(outdir / "influencia_k.csv", index=False)
    plot_k_effect(dfk, outdir)

    # Diretrizes
    print_guidelines(best_k, comp, dfk)

if __name__ == "__main__":
    main()
