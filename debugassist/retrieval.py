from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from debugassist.preprocess import normalize_text


DATA_PATH = Path("data/debug_cases.csv")
TFIDF_PATH = Path("models/tfidf.joblib")


@dataclass
class SimilarCase:
    id: str
    error_family: str
    error_text: str
    fix_text: str
    score: float


class RetrievalIndex:
    def __init__(self, df: pd.DataFrame, vectorizer):
        self.df = df.reset_index(drop=True)
        self.vectorizer = vectorizer

        texts = self.df["error_text"].astype(str).apply(normalize_text).tolist()
        self.matrix = self.vectorizer.transform(texts)

    @classmethod
    def load_default(cls) -> "RetrievalIndex":
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}. Run build_dataset.py first."
            )
            
        if not TFIDF_PATH.exists():
            raise FileNotFoundError(
                f"Vectorizer not found at {TFIDF_PATH}. Run train.py first."
            )

        df = pd.read_csv(DATA_PATH)
        required = {"id", "error_text", "error_family", "fix_text"}
        
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns: {sorted(required)}"
            )

        vectorizer = joblib.load(TFIDF_PATH)
        
        return cls(df=df, vectorizer=vectorizer)

    def query(self, text: str, top_k: int = 3) -> List[SimilarCase]:
        if top_k <= 0:
            return []

        q = normalize_text(text)
        q_vec = self.vectorizer.transform([q])

        sims = cosine_similarity(q_vec, self.matrix).flatten()

        top_idx = sims.argsort()[::-1][:top_k]

        results: List[SimilarCase] = []
        
        for i in top_idx:
            row = self.df.iloc[int(i)]
            results.append(
                SimilarCase(
                    id=str(row["id"]),
                    error_family=str(row["error_family"]),
                    error_text=str(row["error_text"]),
                    fix_text=str(row["fix_text"]),
                    score=float(sims[int(i)]),
                )
            )

        return results
