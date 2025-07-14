# Example pipeline demonstrating synthetic document generation, embedding,
# manual labelling with a Pydantic schema and training classifiers.
# The script mirrors the steps outlined in the repository notebooks but
# adapts them for planning application descriptions.

from __future__ import annotations

import random
from typing import List, Literal

import numpy as np
import pandas as pd

# Optional imports; these will fail if the required libraries are missing.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
except Exception:  # pragma: no cover - sklearn may not be installed
    LogisticRegression = None
    LabelEncoder = None

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch may not be installed
    torch = None
    nn = None

try:
    import joblib
except Exception:  # pragma: no cover - joblib may not be installed
    joblib = None

# ---- Structured output schema for manual labelling ----
from pydantic import BaseModel

class PlanningClassification(BaseModel):
    main_type: Literal[
        "new_home_construction",
        "modification_or_extension",
        "change_of_use",
        "demolition",
        "conditions_or_amendments",
        "tree_works",
        "advertisement_or_signage",
        "unknown",
    ]
    sector: Literal[
        "residential",
        "retail",
        "industrial",
        "agricultural",
        "educational",
        "renewable_energy",
        "infrastructure",
        "hospitality_or_leisure",
        "office",
        "unknown",
    ]

# Toggle network usage. When False, embedding and labelling require a running
# Ollama server and OpenAI-compatible endpoint. When True, the code generates
# random data so the script executes without external dependencies.
simulate = True

EMBED_DIM = 256


def generate_documents(n: int = 50) -> pd.DataFrame:
    """Create a DataFrame of synthetic planning descriptions."""
    templates = [
        "Erection of 2 new houses with garages",
        "Change of use from warehouse to office",
        "Demolition of existing barn and building 3 flats",
        "Installation of rooftop solar panels",
        "Removal of oak tree protected by TPO",
    ]
    docs = [random.choice(templates) + f" #{i}" for i in range(n)]
    return pd.DataFrame({"text": docs, "id": range(n)})


def embed(text: str) -> np.ndarray:
    """Return an embedding vector using ollama or random data."""
    if simulate or torch is None:
        return np.random.rand(EMBED_DIM).astype(np.float32)
    from ollama import Client

    client = Client(host="http://localhost:11434")
    r = client.embeddings(model="mxbai-embed-large", prompt=text)
    return np.array(r["embedding"], dtype=np.float32)


def build_prompt(description: str) -> str:
    return f"""
You are an expert in UK planning law. Classify this planning application:
\n"{description}"\n
Return compact JSON as {{"main_type": "...", "sector": "..."}}
"""


def label_text(text: str) -> PlanningClassification:
    """Label a single text using an LLM or random choices."""
    if simulate:
        main = random.choice(list(PlanningClassification.model_fields["main_type"].annotation.__args__))
        sec = random.choice(list(PlanningClassification.model_fields["sector"].annotation.__args__))
        return PlanningClassification(main_type=main, sector=sec)

    from ollama import Client

    client = Client(host="http://localhost:11434")
    prompt = build_prompt(text)
    response = client.chat(model="llama3.2:3b-instruct-fp16", messages=[{"role": "user", "content": prompt}])
    parsed = PlanningClassification.model_validate_json(response["message"]["content"].strip())
    return parsed


def main() -> None:  # pragma: no cover - example script
    docs = generate_documents(30)
    docs["embedding"] = docs["text"].apply(embed)

    # Randomly sample a subset for manual labelling
    labelled = docs.sample(n=10, random_state=42)
    labels = [label_text(t) for t in labelled["text"]]
    labelled["main_type"] = [lbl.main_type for lbl in labels]
    labelled["sector"] = [lbl.sector for lbl in labels]

    # Prepare feature matrix
    X = np.vstack(labelled["embedding"])

    if LogisticRegression is None or LabelEncoder is None:
        print("scikit-learn not installed; skipping training.")
        return

    le_main = LabelEncoder().fit(labelled["main_type"])
    y_main = le_main.transform(labelled["main_type"])
    clf_main = LogisticRegression(max_iter=200).fit(X, y_main)

    if joblib:
        joblib.dump(clf_main, "main_type_logreg.joblib")
        joblib.dump(le_main, "main_type_encoder.joblib")

    # MLP classifier for sector labels (if torch is available)
    if torch and nn:
        le_sector = LabelEncoder().fit(labelled["sector"])
        y_sec = torch.tensor(le_sector.transform(labelled["sector"]), dtype=torch.long)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        model = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, len(le_sector.classes_)),
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(50):
            opt.zero_grad()
            out = model(X_tensor)
            loss = loss_fn(out, y_sec)
            loss.backward()
            opt.step()

        torch.save(model.state_dict(), "sector_mlp.pt")
        if joblib:
            joblib.dump(le_sector, "sector_encoder.joblib")
    else:
        print("PyTorch not installed; skipped MLP training.")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
