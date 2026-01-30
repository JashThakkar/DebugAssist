from __future__ import annotations

from pathlib import Path

import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from debugassist.preprocess import normalize_text

DATA_PATH = Path("data/debug_cases.csv")
MODELS_DIR = Path("models")
TFIDF_PATH = MODELS_DIR / "tfidf.joblib"
CLF_PATH = MODELS_DIR / "clf.joblib"

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run build_dataset.py first."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if not {"error_text", "error_family"}.issubset(df.columns):
        raise ValueError("CSV must contain 'error_text' and 'error_family' columns")

    X = df["error_text"].astype(str).apply(normalize_text)
    y = df["error_family"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1,
    )

    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"Macro F1: {macro_f1:.3f}")

    joblib.dump(vectorizer, TFIDF_PATH)
    joblib.dump(clf, CLF_PATH)


if __name__ == "__main__":
    main()
