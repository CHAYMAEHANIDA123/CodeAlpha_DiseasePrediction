# === IMPORTATION DES BIBLIOTHÈQUES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)

# === 1. CHARGEMENT DES DONNÉES ===
print("🔄 Chargement du dataset (Breast Cancer Wisconsin)...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
# Dans sklearn: 0 = malignant, 1 = benign
y = pd.Series(data.target, name="target")

print(f"   {X.shape[0]} échantillons, {X.shape[1]} caractéristiques")

# === 2. PRÉTRAITEMENT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 3. ENTRAÎNEMENT ET COMPARAISON DES MODÈLES ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

results = []
fitted_models = {}

print("\n🤖 Entraînement et évaluation des modèles...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({"Modèle": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1-Score": f1, "ROC-AUC": auc})
    fitted_models[name] = {"model": model, "y_pred": y_pred, "y_proba": y_proba}

    print(f"\n📊 {name}")
    print(f"   Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} "
          f"| F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
print("\n📈 Tableau comparatif (trié par F1-Score) :")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Modèle"]
print(f"\n🏆 Meilleur modèle : {best_model_name}")

# === 4. VISUALISATION 1 : COMPARAISON DES MODÈLES ===
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(results_df))
width = 0.15

plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, results_df[metric], width, label=metric)

plt.xlabel("Algorithmes")
plt.ylabel("Score")
plt.title("Comparaison des Performances des Modèles (Breast Cancer Prediction)")
plt.xticks(x + width * 2, results_df["Modèle"])
plt.ylim(0.80, 1.05)
plt.legend(title="Métriques")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# === 5. VISUALISATION 2 : COURBE ROC (meilleur modèle) ===
best = fitted_models[best_model_name]
fpr, tpr, _ = roc_curve(y_test, best["y_proba"])
best_auc = roc_auc_score(y_test, best["y_proba"])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkgreen", label=f"ROC Curve (AUC = {best_auc:.4f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC - Disease Prediction Model (CodeAlpha Task 4)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("disease_roc_curve.png")
plt.close()

# === 6. VISUALISATION 3 : MATRICE DE CONFUSION (meilleur modèle) ===
cm = confusion_matrix(y_test, best["y_pred"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["malignant", "benign"],
            yticklabels=["malignant", "benign"])
plt.title(f"Matrice de Confusion - {best_model_name}")
plt.xlabel("Prédiction")
plt.ylabel("Vérité Terrain")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\n📝 Rapport de classification détaillé (meilleur modèle) :")
print(classification_report(y_test, best["y_pred"], target_names=["malignant (0)", "benign (1)"]))

print("\n✨ Task 4 terminée avec succès !")
print("   Images sauvegardées : model_comparison.png, disease_roc_curve.png, confusion_matrix.png")
