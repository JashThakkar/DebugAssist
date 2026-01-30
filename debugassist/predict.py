from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import typer
import yaml

from debugassist.preprocess import combine_inputs, normalize_text
from debugassist.rules import rule_predict
from debugassist.retrieval import RetrievalIndex, SimilarCase

app = typer.Typer(add_completion=False)

PLAYBOOK_PATH = Path("debugassist/playbooks.yaml")
TFIDF_PATH = Path("models/tfidf.joblib")
CLF_PATH = Path("models/clf.joblib")

LOW_CONF_THRESHOLD = 0.35
TOP_N_ALTERNATIVES = 3


def _load_playbooks(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"playbooks.yaml not found at {path}. Create debugassist/playbooks.yaml first."
        )
        
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        
    if not isinstance(data, dict):
        raise ValueError("playbooks.yaml must parse to a dictionary at the top level.")
    
    return data


def _load_model() -> Tuple[Any, Any]:
    if not TFIDF_PATH.exists() or not CLF_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Run:\n"
            "  1) python3 -m debugassist.build_dataset --total 1200\n"
            "  2) python3 -m debugassist.train\n"
        )
        
    vectorizer = joblib.load(TFIDF_PATH)
    clf = joblib.load(CLF_PATH)
    
    return vectorizer, clf


def _predict_family_ml(
    vectorizer: Any, clf: Any, combined_text: str
) -> Tuple[str, Optional[float], List[Tuple[str, float]]]:

    x = normalize_text(combined_text)
    x_vec = vectorizer.transform([x])

    label = str(clf.predict(x_vec)[0])

    conf: Optional[float] = None
    top_candidates: List[Tuple[str, float]] = []

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x_vec)[0]
        classes = list(clf.classes_)

        pairs = sorted(
            [(str(classes[i]), float(proba[i])) for i in range(len(classes))],
            key=lambda t: t[1],
            reverse=True,
        )
        top_candidates = pairs[:TOP_N_ALTERNATIVES]
        conf = top_candidates[0][1] if top_candidates else None

        if top_candidates:
            label = top_candidates[0][0]

    return label, conf, top_candidates


def _playbook_suggestions(playbooks: Dict[str, Any], family: str, raw_text: str) -> List[str]:
    if family not in playbooks:
        return []

    section = playbooks.get(family, {})
    suggestions: List[str] = []

    checklist = section.get("checklist", [])
    
    if isinstance(checklist, list):
        suggestions.extend([str(x) for x in checklist])

    keyword_tips = section.get("keyword_tips", {})
    
    if isinstance(keyword_tips, dict):
        lowered = raw_text.lower()
        
        for key_phrase, tips in keyword_tips.items():
            key_phrase_str = str(key_phrase)

            if key_phrase_str.lower() in lowered:
                if isinstance(tips, list):
                    suggestions.extend([str(t) for t in tips])

    seen = set()
    out: List[str] = []
    
    for s in suggestions:
        if s not in seen:
            out.append(s)
            seen.add(s)
            
    return out


def _print_header(family: str, method: str, confidence: Optional[float]) -> None:
    typer.echo("")
    typer.echo("       * DebugAssist *")
    typer.echo("==============================")

    if confidence is not None:
        typer.echo(f"\nPredicted family: {family}   (via {method}, confidence={confidence:.2f})")
    else:
        typer.echo(f"\nPredicted family: {family}   (via {method})")


def _print_single_checklist(title: str, suggestions: List[str]) -> None:
    typer.echo(f"\n{title}")
    
    if not suggestions:
        typer.echo("  (No playbook suggestions found for this category yet.)")
        return
    
    for s in suggestions:
        typer.echo(f"  - {s}")


def _print_multi_checklists_for_low_confidence(
    playbooks: Dict[str, Any],
    top_candidates: List[Tuple[str, float]],
    raw_input: str,
    ) -> None:
    
    if not top_candidates:
        typer.echo("\nFix checklist:")
        typer.echo("  (No probability information available.)")
        
        return

    typer.echo("\nLow confidence — showing fix checklists for top candidates:")

    for fam, score in top_candidates:
        suggestions = _playbook_suggestions(playbooks, fam, raw_input)
        title = f"Fix Checklist ({fam}) (score: {score:.2f}):"
        _print_single_checklist(title, suggestions)


def _print_request_exact_error() -> None:
    typer.echo("\nTo improve accuracy, paste the exact error/traceback output from your terminal/IDE.")
    typer.echo("\nExample:")
    typer.echo("  python3 -m debugassist.predict --text \"Traceback (most recent call last): ...\"\n")


def _print_similar_cases(cases: List[SimilarCase]) -> None:
    typer.echo("\nSimilar solved cases:")
    
    if not cases:
        typer.echo("  (No similar cases found.)")
        
        return

    for i, c in enumerate(cases, start=1):
        typer.echo(f"\n  #{i}  score={c.score:.3f}  label={c.error_family}  id={c.id}")
        preview = c.error_text.replace("\n", " ")
        
        if len(preview) > 160:
            preview = preview[:160] + "..."
            
        typer.echo(f"     error: {preview}")

        fix_preview = c.fix_text.strip()
        
        if len(fix_preview) > 200:
            fix_preview = fix_preview[:200] + "..."
            
        typer.echo(f"     fix:   {fix_preview}")
    
    typer.echo("")


@app.command()
def main(
    text: str = typer.Option(..., "--text", "-t", help="Paste the Python error/traceback text."),
    code: Optional[str] = typer.Option(None, "--code", "-c", help="Optional code snippet for added context."),
    top_k: int = typer.Option(3, help="Number of similar cases to show."),
) -> None:

    raw_input = combine_inputs(text, code)

    top_candidates: List[Tuple[str, float]] = []
    method = "rules"
    confidence: Optional[float] = None

    family = rule_predict(text)

    if family is None:
        vectorizer, clf = _load_model()
        family, confidence, top_candidates = _predict_family_ml(vectorizer, clf, raw_input)
        method = "ml"

    playbooks = _load_playbooks(PLAYBOOK_PATH)

    _print_header(family, method, confidence)

    if method == "ml" and confidence is not None and confidence < LOW_CONF_THRESHOLD:
        _print_multi_checklists_for_low_confidence(playbooks, top_candidates, raw_input)

        _print_request_exact_error()
        
        return

    suggestions = _playbook_suggestions(playbooks, family, raw_input)
    _print_single_checklist("Fix checklist:", suggestions)

    retrieval = RetrievalIndex.load_default()
    similar = retrieval.query(raw_input, top_k=top_k)
    _print_similar_cases(similar)


if __name__ == "__main__":
    app()
