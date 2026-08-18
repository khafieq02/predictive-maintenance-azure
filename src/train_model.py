from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

candidates = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
if not candidates:
    raise FileNotFoundError("No CSV or XLSX dataset found in the data folder.")

source = candidates[0]
df = pd.read_csv(source) if source.suffix.lower() == ".csv" else pd.read_excel(source)
df.columns = [str(c).strip() for c in df.columns]

aliases = {
    "Machine failure": "Machine failure",
    "Machine_Failure": "Machine failure",
    "machine failure": "Machine failure",
}
target = next((aliases[c] for c in df.columns if c in aliases), None)
if target is None:
    raise KeyError("Could not find the machine-failure target column.")

X = df.drop(columns=[target]).copy()
y = pd.to_numeric(df[target], errors="coerce").astype(int)

for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype(str).str.strip()

categorical = X.select_dtypes(include="object").columns.tolist()
numeric = X.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer(
    [
        ("numeric", SimpleImputer(strategy="median"), numeric),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical,
        ),
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
pipeline = Pipeline([("preprocess", preprocess), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "model_name": "RandomForestClassifier",
    "split_configuration": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    },
    "class_imbalance_method": "class_weight=balanced",
    "test_records": int(len(y_test)),
    "test_failures": int(y_test.sum()),
}

with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

results = X_test.copy()
results["Actual_Machine_Failure"] = y_test.to_numpy()
results["Predicted_Machine_Failure"] = pred
results["Failure_Risk_Probability"] = np.round(prob, 6)
results.to_csv(OUTPUT_DIR / "predictions_for_powerbi.csv", index=False)

feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
importance = pipeline.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
importance_df = importance_df.sort_values("Importance", ascending=False)
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

plot_df = importance_df.head(15).sort_values("Importance")
plt.figure(figsize=(8, 6))
sns.barplot(data=plot_df, x="Importance", y="Feature", color="#2196F3")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
plt.close()

joblib.dump(pipeline, OUTPUT_DIR / "random_forest_pipeline.joblib")
print(json.dumps(metrics, indent=2))
