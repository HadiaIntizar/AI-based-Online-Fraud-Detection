# """
# Online Fraud Detection Using AI - Training Pipeline

# This script follows the ML project guidelines:
# 1. Load dataset
# 2. Preprocess data
# 3. Extract/select features
# 4. Train 10 models
# 5. Evaluate all models
# 6. Generate visuals
# 7. Save the best model as .pkl
# 8. Generate a short final report
# """

# from __future__ import annotations

# import json
# import warnings
# from pathlib import Path

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.base import clone
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import (
#     AdaBoostClassifier,
#     ExtraTreesClassifier,
#     GradientBoostingClassifier,
#     RandomForestClassifier,
# )
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     ConfusionMatrixDisplay,
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
#     roc_curve,
# )
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.svm import LinearSVC
# from sklearn.tree import DecisionTreeClassifier

# warnings.filterwarnings("ignore")
# sns.set_theme(style="whitegrid")

# ROOT = Path(__file__).resolve().parent
# DATA_PATH = ROOT / "data" / "account_profiles.csv"
# MODEL_DIR = ROOT / "models"
# VISUAL_DIR = ROOT / "visuals"
# REPORT_DIR = ROOT / "reports"
# TARGET = "is_fraudster"
# ID_COLUMNS = ["account_id"]
# RANDOM_STATE = 42

# MODEL_DIR.mkdir(exist_ok=True)
# VISUAL_DIR.mkdir(exist_ok=True)
# REPORT_DIR.mkdir(exist_ok=True)


# def safe_auc(y_true, y_score):
#     try:
#         return roc_auc_score(y_true, y_score)
#     except Exception:
#         return np.nan


# def get_scores(model, X_test):
#     """Return probability-like fraud scores for ROC-AUC and risk display."""
#     if hasattr(model, "predict_proba"):
#         return model.predict_proba(X_test)[:, 1]
#     if hasattr(model, "decision_function"):
#         raw = model.decision_function(X_test)
#         # Min-max scale decision scores to 0..1 for visualization/API confidence.
#         return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
#     preds = model.predict(X_test)
#     return preds.astype(float)


# def load_and_clean_data() -> pd.DataFrame:
#     print(f"Loading dataset: {DATA_PATH}")
#     df = pd.read_csv(DATA_PATH)
#     print(f"Original shape: {df.shape}")

#     df = df.drop_duplicates()
#     print(f"After duplicate removal: {df.shape}")

#     if TARGET not in df.columns:
#         raise ValueError(f"Target column '{TARGET}' not found in dataset.")

#     # Ensure binary integer target.
#     df[TARGET] = df[TARGET].astype(int)
#     return df


# def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
#     numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
#     categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

#     numeric_pipeline = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler()),
#         ]
#     )
#     categorical_pipeline = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("encoder", OneHotEncoder(handle_unknown="ignore")),
#         ]
#     )

#     return ColumnTransformer(
#         transformers=[
#             ("num", numeric_pipeline, numeric_features),
#             ("cat", categorical_pipeline, categorical_features),
#         ],
#         remainder="drop",
#     )


# def define_models():
#     return {
#         "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
#         "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=RANDOM_STATE),
#         "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
#         "Linear SVM": LinearSVC(class_weight="balanced", max_iter=2500, random_state=RANDOM_STATE),
#         "KNN": KNeighborsClassifier(n_neighbors=7),
#         "Naive Bayes": GaussianNB(),
#         "AdaBoost": AdaBoostClassifier(n_estimators=120, learning_rate=0.8, random_state=RANDOM_STATE),
#         "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, random_state=RANDOM_STATE),
#         "Extra Trees": ExtraTreesClassifier(n_estimators=150, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
#         "ANN / MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=120, early_stopping=True, random_state=RANDOM_STATE),
#     }


# def plot_target_distribution(df: pd.DataFrame):
#     plt.figure(figsize=(7, 5))
#     ax = sns.countplot(data=df, x=TARGET, palette=["#1f77b4", "#d62728"])
#     ax.set_title("Target Distribution: Non-Fraudster vs Fraudster", fontsize=14, weight="bold")
#     ax.set_xlabel("is_fraudster (0 = No, 1 = Yes)")
#     ax.set_ylabel("Number of Accounts")
#     for container in ax.containers:
#         ax.bar_label(container)
#     plt.tight_layout()
#     plt.savefig(VISUAL_DIR / "01_target_distribution.png", dpi=200)
#     plt.close()


# def plot_correlation_heatmap(df: pd.DataFrame):
#     numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors="ignore")
#     # Use top variables most correlated with the target to keep graph readable.
#     correlations = df.select_dtypes(include=[np.number]).corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
#     top_cols = correlations.index[:14].tolist()
#     corr_df = df[top_cols].corr(numeric_only=True)

#     plt.figure(figsize=(12, 9))
#     sns.heatmap(corr_df, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
#     plt.title("Correlation Heatmap of Important Numerical Features", fontsize=14, weight="bold")
#     plt.tight_layout()
#     plt.savefig(VISUAL_DIR / "02_correlation_heatmap.png", dpi=200)
#     plt.close()


# def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
#     results = []
#     trained_models = {}
#     roc_data = {}

#     for name, estimator in define_models().items():
#         print(f"Training: {name}")
#         # GaussianNB needs dense array after preprocessing.
#         if name == "Naive Bayes":
#             model = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("to_dense", DenseTransformer()), ("classifier", estimator)])
#         else:
#             model = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("classifier", estimator)])

#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         scores = get_scores(model, X_test)

#         metrics = {
#             "Model": name,
#             "Accuracy": accuracy_score(y_test, preds),
#             "Precision": precision_score(y_test, preds, zero_division=0),
#             "Recall": recall_score(y_test, preds, zero_division=0),
#             "F1-Score": f1_score(y_test, preds, zero_division=0),
#             "ROC-AUC": safe_auc(y_test, scores),
#         }
#         results.append(metrics)
#         trained_models[name] = model

#         fpr, tpr, _ = roc_curve(y_test, scores)
#         roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": metrics["ROC-AUC"]}

#         print(f"  F1={metrics['F1-Score']:.4f}, ROC-AUC={metrics['ROC-AUC']:.4f}")

#     results_df = pd.DataFrame(results).sort_values(by=["F1-Score", "ROC-AUC", "Accuracy"], ascending=False)
#     return results_df, trained_models, roc_data


# class DenseTransformer:
#     """Small sklearn-compatible transformer to convert sparse matrices to dense arrays."""

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         return X.toarray() if hasattr(X, "toarray") else X


# def plot_model_comparison(results_df: pd.DataFrame):
#     results_melted = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], var_name="Metric", value_name="Score")
#     plt.figure(figsize=(14, 7))
#     sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric")
#     plt.title("Model Performance Comparison", fontsize=15, weight="bold")
#     plt.xticks(rotation=35, ha="right")
#     plt.ylim(0, 1.05)
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig(VISUAL_DIR / "03_model_comparison.png", dpi=200)
#     plt.close()


# def plot_roc_curves(roc_data: dict):
#     plt.figure(figsize=(10, 8))
#     for name, data in roc_data.items():
#         plt.plot(data["fpr"], data["tpr"], label=f"{name} (AUC={data['auc']:.3f})", linewidth=1.8)
#     plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curves for Fraud Detection Models", fontsize=14, weight="bold")
#     plt.legend(fontsize=8)
#     plt.tight_layout()
#     plt.savefig(VISUAL_DIR / "04_roc_curves.png", dpi=200)
#     plt.close()


# def plot_confusion_matrix(best_model_name, best_model, X_test, y_test):
#     preds = best_model.predict(X_test)
#     cm = confusion_matrix(y_test, preds)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
#     disp.plot(cmap="Blues", values_format="d")
#     plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, weight="bold")
#     plt.tight_layout()
#     plt.savefig(VISUAL_DIR / "05_best_model_confusion_matrix.png", dpi=200)
#     plt.close()


# def plot_feature_importance(best_model, X: pd.DataFrame):
#     classifier = best_model.named_steps["classifier"]
#     preprocessor = best_model.named_steps["preprocessor"]

#     try:
#         feature_names = preprocessor.get_feature_names_out()
#     except Exception:
#         feature_names = np.array(X.columns)

#     importances = None
#     if hasattr(classifier, "feature_importances_"):
#         importances = classifier.feature_importances_
#     elif hasattr(classifier, "coef_"):
#         importances = np.abs(classifier.coef_[0])

#     if importances is None:
#         # Model-agnostic fallback: correlation with target is already provided in heatmap.
#         return

#     n = min(len(feature_names), len(importances))
#     fi = pd.DataFrame({"Feature": feature_names[:n], "Importance": importances[:n]})
#     fi["Feature"] = fi["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
#     fi = fi.sort_values("Importance", ascending=False).head(15)

#     plt.figure(figsize=(10, 7))
#     sns.barplot(data=fi, x="Importance", y="Feature", palette="viridis")
#     plt.title("Top 15 Important Features", fontsize=14, weight="bold")
#     plt.tight_layout()
#     plt.savefig(VISUAL_DIR / "06_feature_importance.png", dpi=200)
#     plt.close()


# def plot_fraud_risk_analysis(df: pd.DataFrame):
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#     sns.boxplot(data=df, x=TARGET, y="risk_score", ax=axes[0, 0], palette="Set2")
#     axes[0, 0].set_title("Risk Score by Fraudster Class")

#     sns.boxplot(data=df, x=TARGET, y="avg_ip_risk", ax=axes[0, 1], palette="Set2")
#     axes[0, 1].set_title("Average IP Risk by Fraudster Class")

#     sns.boxplot(data=df, x=TARGET, y="pct_foreign", ax=axes[1, 0], palette="Set2")
#     axes[1, 0].set_title("Foreign Transaction Percentage by Class")

#     sns.countplot(data=df, x="account_type", hue=TARGET, ax=axes[1, 1], palette=["#1f77b4", "#d62728"])
#     axes[1, 1].set_title("Account Type vs Fraudster Class")
#     axes[1, 1].legend(title="is_fraudster")

#     plt.suptitle("Fraud Risk Exploratory Visual Dashboard", fontsize=16, weight="bold")
#     plt.tight_layout()
#     plt.savefig(VISUAL_DIR / "07_fraud_risk_dashboard.png", dpi=200)
#     plt.close()


# def dataframe_to_markdown(df: pd.DataFrame) -> str:
#     """Create a markdown table without requiring the optional tabulate package."""
#     display_df = df.copy()
#     for col in display_df.select_dtypes(include=[np.number]).columns:
#         display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
#     headers = list(display_df.columns)
#     lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
#     for _, row in display_df.iterrows():
#         lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
#     return "\n".join(lines)


# def generate_report(df, results_df, best_name, best_model, X_test, y_test):
#     best_preds = best_model.predict(X_test)
#     report_text = classification_report(y_test, best_preds, target_names=["Not Fraudster", "Fraudster"])
#     results_markdown = dataframe_to_markdown(results_df)

#     content = f"""# Final Report: Online Fraud Detection Using AI

# ## Abstract
# This project develops an AI-based online fraud detection system using account profile and transaction behavior data. The system classifies whether an account is likely to be a fraudster.

# ## Dataset
# - Records: {len(df):,}
# - Features before preprocessing: {df.shape[1] - 1}
# - Target: `{TARGET}`
# - Non-fraudster accounts: {(df[TARGET] == 0).sum():,}
# - Fraudster accounts: {(df[TARGET] == 1).sum():,}

# ## Methodology
# 1. Removed duplicate records.
# 2. Handled missing values using median/mode imputation.
# 3. Encoded categorical features using one-hot encoding.
# 4. Scaled numerical features using StandardScaler.
# 5. Split data into 80% training and 20% testing.
# 6. Trained 10 ML/AI models.
# 7. Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

# ## Model Comparison

# {results_markdown}

# ## Best Model
# The selected best model is **{best_name}** because it achieved the strongest combined F1-score, ROC-AUC, and accuracy on the test set.

# ## Best Model Classification Report

# ```text
# {report_text}
# ```

# ## Graphical Analysis Files
# - `visuals/01_target_distribution.png`
# - `visuals/02_correlation_heatmap.png`
# - `visuals/03_model_comparison.png`
# - `visuals/04_roc_curves.png`
# - `visuals/05_best_model_confusion_matrix.png`
# - `visuals/06_feature_importance.png`
# - `visuals/07_fraud_risk_dashboard.png`

# ## Deployment
# The saved model is used by `app.py`, a Flask web API and frontend application.

# ## Conclusion
# The project satisfies the DS/AI workflow from dataset collection to model training, evaluation, visual analysis, saved model, API, and frontend integration.
# """
#     (REPORT_DIR / "final_report.md").write_text(content, encoding="utf-8")


# def main():
#     df = load_and_clean_data()
#     plot_target_distribution(df)
#     plot_correlation_heatmap(df)
#     plot_fraud_risk_analysis(df)

#     y = df[TARGET]
#     X = df.drop(columns=[TARGET] + [col for col in ID_COLUMNS if col in df.columns])

#     with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
#         json.dump(X.columns.tolist(), f, indent=2)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
#     )

#     preprocessor = build_preprocessor(X)
#     results_df, trained_models, roc_data = evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
#     results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)
#     print("\nModel Results:")
#     print(results_df.to_string(index=False))

#     best_name = results_df.iloc[0]["Model"]
#     best_model = trained_models[best_name]

#     joblib.dump(best_model, MODEL_DIR / "best_fraud_model.pkl")
#     with open(MODEL_DIR / "best_model_info.json", "w", encoding="utf-8") as f:
#         json.dump({"best_model": best_name, "target": TARGET, "metrics": results_df.iloc[0].to_dict()}, f, indent=2)

#     plot_model_comparison(results_df)
#     plot_roc_curves(roc_data)
#     plot_confusion_matrix(best_name, best_model, X_test, y_test)
#     plot_feature_importance(best_model, X)
#     generate_report(df, results_df, best_name, best_model, X_test, y_test)

#     print(f"\nBest Model: {best_name}")
#     print(f"Saved model: {MODEL_DIR / 'best_fraud_model.pkl'}")
#     print(f"Saved visuals in: {VISUAL_DIR}")
#     print(f"Saved report: {REPORT_DIR / 'final_report.md'}")


# if __name__ == "__main__":
#     main()
