# DebugAssist 🛠️
A Local ML-Powered Python Debugging Assistant

## Overview
DebugAssist is a local, Python based debugging assistant that helps developers understand and troubleshoot programming errors by analyzing error messages, stack traces, and natural-language descriptions of bugs.

Users paste an error (or describe what went wrong), and DebugAssist predicts the most likely **error family** (for example: type errors, value errors, key errors). Based on this prediction, the tool provides structured debugging guidance using curated fix playbooks and, when appropriate, related historical examples.

The entire system runs locally, requires no deployment, and demonstrates a complete machine-learning workflow from data generation to inference.

## What This Project Does
DebugAssist combines deterministic logic with machine learning to handle real-world debugging scenarios:

- Uses rule-based detection to immediately identify obvious, high-confidence error patterns
- Falls back to a machine-learning classifier when input is incomplete, noisy, or written in natural language
- Applies confidence-aware behavior to avoid misleading results when predictions are uncertain
- Provides actionable fix checklists using YAML-based playbooks
- Uses similarity-based retrieval to surface related errors and known solutions when confidence is high

This hybrid design mirrors how real developer tools balance reliability and flexibility.

## Tech Stack

### Language
- Python 

### Machine Learning
- scikit-learn
- TF-IDF vectorization (word n-grams)
- Logistic Regression (multiclass text classification)

### Data
- Synthetic dataset generation
- Configurable dataset size
- Balanced class distribution

### Tooling
- Typer (command-line interface)
- PyYAML (playbook configuration)
- joblib (model persistence)

### Core Concepts Demonstrated
- Text preprocessing and normalization
- Supervised ML classification
- Confidence-based inference logic
- Cosine similarity retrieval
- Hybrid rules + ML system design


# Execution Flow (How the Project Runs)

This section explains what happens internally as each command is executed, from environment setup to running a prediction.
(Check howToRunDBA.txt for more info)

## 1. Virtual Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
```

A Python virtual environment is created to isolate project dependencies.


## 2. Dependency Installation

```bash
pip install -r requirements.txt
```

All required libraries are installed, including:

- **Machine learning**
  - `scikit-learn`
- **CLI tooling**
  - `Typer`
- **Configuration and persistence**
  - `PyYAML`
  - `joblib`

At this point, the environment is fully prepared to run the project.

---

## 3. Dataset Generation

```bash
python3 -m debugassist.build_dataset --total 1200
```

or

```bash
python3 -m debugassist.build_dataset --per-class 120
```

This step creates a **synthetic training dataset of programming errors**.

### What happens internally

- Predefined error templates are used for each error family
- Each dataset entry includes:
  - An error message (traceback-style or natural language)
  - A labeled error family
  - A corresponding fix description
- The dataset is balanced across classes unless specified otherwise

### Output

The final dataset is written to:

```
data/debug_cases.csv
```

This approach allows controlled experimentation without relying on real production logs.

---

## 4. Model Training

```bash
python3 -m debugassist.train
```

### Training workflow

- The dataset is loaded from `data/debug_cases.csv`
- Error text is normalized and vectorized using **TF-IDF**
- A **Logistic Regression** classifier is trained to predict error families

### Saved artifacts

```
models/tfidf.joblib
models/clf.joblib
```

### Evaluation

- Training metrics such as **precision** and **F1 score** are printed to the console
- These metrics help validate model performance

---

## 5. Running a Prediction (Rules + ML)

```bash
python3 -m debugassist.predict --text "<error or description>"
```

When a prediction is executed, the system follows a layered decision process:

### 1. Rule-Based Evaluation

- The input is checked against high-confidence rules
- If a rule matches, the corresponding error family is returned immediately

### 2. Machine Learning Fallback

- If no rule matches:
  - The input is vectorized using the trained TF-IDF model
  - The ML classifier predicts the most likely error family
  - A confidence score is calculated

### 3. Confidence-Aware Output

**High confidence prediction:**
- A focused fix checklist is shown
- Similar solved cases are displayed

**Low confidence prediction:**
- Fix checklists for the top three predicted families are shown
- Similar-case retrieval is skipped
- The user is prompted to provide the exact traceback

### 4. Playbook Guidance

- Fix suggestions are pulled from `playbooks.yaml`
- Keyword-based tips are added when applicable

This layered approach ensures accuracy while avoiding misleading recommendations.

---

## 6. Testing and Iteration

After the system is running, it can be continuously improved:

- Increase dataset size to improve ML accuracy
- Retrain models with new data
- Test natural-language inputs to evaluate generalization
- Refine rule logic independently of ML
- Update fix playbooks without retraining models

This design makes **DebugAssist** easy to iterate on, extend, and maintain.
