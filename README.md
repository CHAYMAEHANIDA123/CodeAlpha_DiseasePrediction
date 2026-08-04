# Breast Cancer Prediction Model — CodeAlpha ML Internship (Task 4)

## 📌 Objectif
Prédire si une tumeur est **maligne** ou **bénigne** à partir de données médicales structurées, dans le cadre de la Task 4 du stage Machine Learning chez CodeAlpha ("Disease Prediction from Medical Data").

## 📊 Dataset
- **Breast Cancer Wisconsin (Diagnostic)** — disponible via `sklearn.datasets.load_breast_cancer`
- 569 échantillons, 30 caractéristiques (rayon, texture, périmètre, aire, etc.)
- Cible : 0 = maligne (malignant), 1 = bénigne (benign)

## ⚙️ Méthodologie
1. Chargement et exploration des données
2. Séparation train/test (80/20), stratifiée
3. Mise à l'échelle des caractéristiques (`StandardScaler`)
4. Entraînement et comparaison de 4 modèles de classification :
   - Logistic Regression
   - SVM (`probability=True`)
   - Random Forest (100 arbres)
   - XGBoost
5. Évaluation : Accuracy, Precision, Recall, F1-Score, ROC-AUC
6. Sélection du meilleur modèle selon le F1-Score

## 📈 Résultats

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** 🏆 | 0.9825 | 0.9861 | 0.9861 | **0.9861** | 0.9954 |
| SVM | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9939 |
| XGBoost | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9901 |

Le modèle **Logistic Regression** obtient les meilleures performances globales (F1-Score : 0.9861, ROC-AUC : 0.9954).

📊 Comparaison visuelle : voir `model_comparison.png`
📉 Courbe ROC : voir `disease_roc_curve.png`
🔢 Matrice de confusion : voir `confusion_matrix.png`

## 🛠️ Stack technique
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn

## 🚀 Utilisation
```bash
pip install -r requirements.txt
python disease_prediction.py
```

## 📁 Structure du dépôt
