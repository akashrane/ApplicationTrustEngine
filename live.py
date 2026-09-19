from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from integrity import INJECTION_RE, inspect_pdf, sanitize
from parser import HybridResumeParser
from provenance import analyze_candidate, fetch_github
from score import anomaly_raw, percentile, qualification_score, quadrant, trust_confidence, verification_action
from shared.schema import SKILL_VOCABULARY

ROOT = Path(__file__).parent
MODELS = ROOT / "data/models"
PARSER = HybridResumeParser()


def embed_one(text: str) -> np.ndarray:
    meta = joblib.load(MODELS / "embedder.joblib")
    if meta["choice"] == "sentence_transformer":
        from sentence_transformers import SentenceTransformer
        return np.asarray(SentenceTransformer(meta["model_name"], local_files_only=True).encode([text], normalize_embeddings=True))
    vectorizer = joblib.load(MODELS / "tfidf.joblib")
    svd = joblib.load(MODELS / "svd.joblib")
    value = svd.transform(vectorizer.transform([text]))
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, 1e-12)


def github_evidence(username: str) -> dict | None:
    if not username:
        return None
    cache_dir = ROOT / "data/raw/github_live"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{re.sub(r'[^\w.-]', '_', username)}.json"
    try:
        evidence = fetch_github(username, os.getenv("GITHUB_TOKEN", ""))
        cache_path.write_text(json.dumps(evidence, indent=2))
        return evidence
    except Exception as exc:
        print(f"GitHub lookup failed for @{username}: {exc}")
        return json.loads(cache_path.read_text()) if cache_path.exists() else None


def analyze_pdf(path: str | Path, github_override: str = "", force_llm: bool = False) -> dict:
    load_dotenv(ROOT / ".env")
    path = Path(path)
    hidden, pdf_text = inspect_pdf(path)
    
    # Extract embedded hyperlink annotations (e.g. clickable icon links)
    doc_links = []
    try:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            for page in doc:
                for link in page.get_links():
                    uri = link.get("uri")
                    if uri:
                        doc_links.append(uri)
    except Exception:
        pass

    full_text = pdf_text + ("\nLINKS:\n" + "\n".join(doc_links) if doc_links else "")
    sanitized = sanitize(full_text)
    override = ROOT / "data/raw/my_resume.json"
    if override.exists():
        record = json.loads(override.read_text())
        record.setdefault("candidate_id", "LIVE-OVERRIDE")
        record.setdefault("submitted_at", datetime.now(timezone.utc).isoformat())
        record["text"] = sanitized
    else:
        record = PARSER.parse_text(sanitized, source_name=path.name, force_llm=force_llm)
    
    extracted_gh = record.get("github_username", "")
    if github_override and github_override.strip():
        record["github_username"] = github_override.strip()
        record["github_source"] = "manual_override"
    elif extracted_gh:
        record["github_source"] = "auto_extracted"
    else:
        record["github_source"] = "none"

    injection_matches = [m.group(0).strip() for m in INJECTION_RE.finditer(pdf_text)]
    events = []
    for item in hidden:
        event_type = "prompt_injection" if INJECTION_RE.search(item["text"]) else "hidden_text"
        events.append({"type": event_type, "severity": "high" if event_type == "prompt_injection" else "medium", "text": item["text"], "reason": item["reason"]})
    integrity = {"hidden_spans": hidden, "injection_detected": bool(injection_matches), "security_events": events}
    embedding = embed_one(sanitized)
    reducer10 = joblib.load(MODELS / "umap10.joblib"); reducer2 = joblib.load(MODELS / "umap2.joblib")
    point10, point2 = reducer10.transform(embedding)[0], reducer2.transform(embedding)[0]
    training = joblib.load(MODELS / "training_embeddings.joblib")
    ids = joblib.load(MODELS / "training_ids.joblib")
    similarities = cosine_similarity(embedding, training)[0]
    order = np.argsort(similarities)[::-1][:3]
    geometry = joblib.load(MODELS / "cluster_geometry.joblib")
    choices = [(label, float(np.linalg.norm(point10 - center))) for label, center in geometry["centroids"].items()]
    label, distance = min(choices, key=lambda item: item[1]) if choices else (-1, float("inf"))
    if label == -1 or distance > geometry["radii"].get(label, 0):
        label = -1
    population = json.loads((ROOT / "data/processed/population.json").read_text())
    cluster = next((c for c in population["clusters"] if c["cluster"] == label), None)
    coordination = cluster["coordination_score"] if cluster else 0
    evidence = github_evidence(record.get("github_username", ""))
    prov = analyze_candidate(record, evidence)
    q_score, requirements = qualification_score(sanitized)
    mostly_unverified = sum(c["status"] == "unverified" for c in prov["claims"]) >= len(prov["claims"]) * 0.75
    raw = anomaly_raw(float(similarities.max()), len(hidden), prov["contradiction_count"], prov["github_account_age_days"], prov["P"], mostly_unverified)
    scored = json.loads((ROOT / "data/processed/scored.json").read_text())
    anomaly = percentile(raw, scored["anomaly_baseline"])
    confidence, trust_score = trust_confidence(prov["P"], anomaly, coordination, prov["claims"])
    quad = quadrant(q_score, trust_score, confidence)
    action = verification_action(prov, integrity, coordination)
    names = {c["candidate_id"]: c["name"] for c in scored["candidates"]}
    return {
        **record, "Q": q_score, "P": prov["P"], "A": anomaly, "C": coordination, "requirements": requirements,
        "claims": prov["claims"], "github_timeline": prov["github_timeline"], "github_account_age_days": prov["github_account_age_days"],
        "contradiction_count": prov["contradiction_count"], "hidden_spans": hidden, "injection_detected": bool(injection_matches),
        "security_events": events, "x": float(point2[0]), "y": float(point2[1]), "cluster": label,
        "cluster_name": cluster["display_name"] if cluster else "Noise", "nn_max_similarity": round(float(similarities.max()), 4),
        "trust_confidence": confidence, "trust_score": trust_score, "quadrant": quad, "verification_action": action,
        "similar_candidates": [{"candidate_id": ids[i], "name": names.get(ids[i], ids[i]), "similarity": round(float(similarities[i]), 3)} for i in order],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--github", default="")
    args = parser.parse_args()
    result = analyze_pdf(args.pdf, args.github)
    print(json.dumps({k: result[k] for k in ["candidate_id", "name", "Q", "P", "A", "C", "quadrant", "verification_action", "injection_detected", "similar_candidates"]}, indent=2))
