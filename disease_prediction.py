# === IMPORTATION DES BIBLIOTHÈQUES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report

# === 1. CHARGEMENT DES DONNÉES ===
print("🔄 Chargement du dataset (Diabetes)...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# === 2. PRÉTRAITEMENT (FEATURE ENGINEERING) ===
print("🛠️ Prétraitement des données...")
# Dans ce dataset, certaines valeurs manquantes sont codées par 0 (ex: Glucose, BMI). On les remplace par NaN.
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Imputation des valeurs manquantes par la médiane
imputer = SimpleImputer(strategy='median')
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

# Séparation des features (X) et de la cible (y = Outcome)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Séparation Train/Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Mise à l'échelle (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 3. ENTRAÎNEMENT DU MODÈLE (RANDOM FOREST) ===
print("🤖 Entraînement du modèle Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# === 4. PRÉDICTION ET ÉVALUATION ===
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n📈 Résultats du Modèle (Disease Prediction) :")
print(f"✅ Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"🎯 Precision : {precision_score(y_test, y_pred):.4f}")
print(f"🔍 Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"⚖️ F1-Score   : {f1_score(y_test, y_pred):.4f}")
print(f"📉 ROC-AUC   : {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n📝 Rapport de classification détaillé :")
print(classification_report(y_test, y_pred, target_names=['No Diabetes (0)', 'Diabetes (1)']))

# === 5. VISUALISATION (ROC CURVE) ===
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkgreen', label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC - Disease Prediction Model (CodeAlpha Task 4)')
plt.legend(loc='lower right')
plt.savefig('disease_roc_curve.png') # Sauvegarde de l'image pour LinkedIn
plt.show()

print("\n✨ Task 4 Terminée avec succès! L'image 'disease_roc_curve.png' a été sauvegardée.")