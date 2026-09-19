from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import umap
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from shared.schema import RANDOM_STATE

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "data/models"


def load_inputs() -> tuple[list[dict], dict[str, dict]]:
    resumes = json.loads((ROOT / "data/raw/resumes.json").read_text())
    integrity = {x["candidate_id"]: x for x in json.loads((ROOT / "data/processed/integrity.json").read_text())}
    return resumes, integrity


def embed_texts(texts: list[str]) -> tuple[np.ndarray, dict]:
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # The core pipeline is offline: use MiniLM when already cached, otherwise fail fast to TF-IDF/SVD.
        model = SentenceTransformer(model_name, local_files_only=True)
        values = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        print(f"Embedding method: {model_name}")
        return np.asarray(values), {"choice": "sentence_transformer", "model_name": model_name}
    except Exception as exc:
        print(f"Sentence transformer unavailable ({type(exc).__name__}); using TF-IDF + SVD")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(128, matrix.shape[1] - 1), random_state=RANDOM_STATE)
        values = normalize(svd.fit_transform(matrix))
        joblib.dump(vectorizer, MODEL_DIR / "tfidf.joblib")
        joblib.dump(svd, MODEL_DIR / "svd.joblib")
        return np.asarray(values), {"choice": "tfidf_svd", "model_name": "TF-IDF (1,2) + SVD-128"}


def base_domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    pieces = host.split(".")
    return ".".join(pieces[-2:]) if len(pieces) >= 2 else host


def word_ngrams(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def densest_window(times: list[datetime], minutes: int = 60) -> int:
    ordered = sorted(times)
    best = left = 0
    for right, value in enumerate(ordered):
        while value - ordered[left] > timedelta(minutes=minutes):
            left += 1
        best = max(best, right - left + 1)
    return best


def cluster_fingerprint(label: int, indices: list[int], candidates: list[dict], embeddings: np.ndarray, baseline: float) -> dict:
    members = [candidates[i] for i in indices]
    local = cosine_similarity(embeddings[indices])
    upper = local[np.triu_indices_from(local, k=1)]
    mean_sim = float(np.mean(upper)) if len(upper) else 0.0
    grams = [word_ngrams(c["sanitized_text"]) for c in members]
    counts = Counter(g for doc in grams for g in doc)
    shared = {g for g, count in counts.items() if count >= len(members) * 0.5}
    shared_ratio = float(np.mean([len(doc & shared) / max(1, len(doc)) for doc in grams]))
    times = [datetime.fromisoformat(c["submitted_at"].replace("Z", "+00:00")) for c in members]
    dense_count = densest_window(times)
    time_ratio = dense_count / len(members)
    domains = [base_domain(c.get("portfolio_url", "")) for c in members if c.get("portfolio_url")]
    common_domain, domain_count = (Counter(domains).most_common(1)[0] if domains else ("none", 0))
    domain_ratio = domain_count / len(members)
    all_skills = [s for c in members for s in c.get("skills", [])]
    skill_diversity = len({s.casefold() for s in all_skills}) / max(1, len(all_skills))
    similarity_gate = max(baseline + 0.18, 0.72)
    similarity_signal = float(np.clip((mean_sim - similarity_gate) / max(0.01, 0.95 - similarity_gate), 0, 1))
    score = round(100 * (0.35 * similarity_signal + 0.25 * shared_ratio + 0.25 * time_ratio + 0.15 * domain_ratio))
    if mean_sim <= similarity_gate:
        score = 0
    tags = Counter(c["seed_tag"] for c in members)
    seed = tags.most_common(1)[0][0]
    names = {"cluster_a": "Cluster A · shared portfolio", "cluster_b": "Cluster B · synchronized template", "cluster_c": "Cluster C · bootcamp cohort"}
    display = names.get(seed, f"Natural cohort {label}")
    summary = (f"{len(members)} résumés with {mean_sim:.0%} similar wording; {dense_count} submitted within the densest 60-minute window; "
               f"{domain_count} share {common_domain}. Possible benign explanations include a shared template, bootcamp, résumé service, or coordinated applicants.")
    return {
        "cluster": label, "display_name": display, "size": len(members), "mean_similarity": round(mean_sim, 3),
        "shared_phrase_ratio": round(shared_ratio, 3), "submission_concentration": round(time_ratio, 3),
        "densest_60m_count": dense_count, "shared_portfolio_domain_ratio": round(domain_ratio, 3),
        "common_domain": common_domain, "skill_diversity": round(skill_diversity, 3), "coordination_score": score,
        "summary": summary, "seed_mix": dict(tags), "member_ids": [c["candidate_id"] for c in members],
        "shared_phrases": sorted(shared, key=lambda g: (-counts[g], g))[:16],
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    resumes, integrity = load_inputs()
    candidates = []
    for rec in resumes:
        check = integrity[rec["candidate_id"]]
        if check["duplicate_of"]:
            continue
        candidates.append({**rec, "sanitized_text": check["sanitized_text"]})
    texts = [c["sanitized_text"] for c in candidates]
    embeddings, embed_meta = embed_texts(texts)
    reducer10 = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.05, metric="cosine", random_state=RANDOM_STATE, transform_seed=RANDOM_STATE, n_jobs=1)
    reducer2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05, metric="cosine", random_state=RANDOM_STATE, transform_seed=RANDOM_STATE, n_jobs=1)
    reduced10 = reducer10.fit_transform(embeddings)
    reduced2 = reducer2.fit_transform(embeddings)
    labels = HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean", cluster_selection_method="eom", copy=True).fit_predict(reduced10)
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, -1)
    nearest = sims.max(axis=1)
    sample = cosine_similarity(embeddings[:min(400, len(embeddings))])
    baseline = float(np.median(sample[np.triu_indices_from(sample, k=1)]))
    clusters = []
    for label in sorted(set(labels) - {-1}):
        idx = np.where(labels == label)[0].tolist()
        clusters.append(cluster_fingerprint(int(label), idx, candidates, embeddings, baseline))
    fp_map = {c["cluster"]: c for c in clusters}
    points = []
    for i, candidate in enumerate(candidates):
        label = int(labels[i])
        points.append({
            "candidate_id": candidate["candidate_id"], "x": round(float(reduced2[i, 0]), 5), "y": round(float(reduced2[i, 1]), 5),
            "cluster": label, "cluster_name": fp_map.get(label, {}).get("display_name", "Noise"),
            "nn_max_similarity": round(float(nearest[i]), 4), "C": fp_map.get(label, {}).get("coordination_score", 0),
        })
    centroids, radii = {}, {}
    for label in set(labels) - {-1}:
        values = reduced10[labels == label]
        center = values.mean(axis=0)
        distances = np.linalg.norm(values - center, axis=1)
        centroids[int(label)] = center
        radii[int(label)] = float(np.quantile(distances, 0.90) * 1.25)
    output = {"embedding_method": embed_meta["model_name"], "baseline_similarity": round(baseline, 4), "candidates": points, "clusters": clusters}
    (ROOT / "data/processed/population.json").write_text(json.dumps(output, indent=2))
    joblib.dump(embed_meta, MODEL_DIR / "embedder.joblib")
    joblib.dump(reducer10, MODEL_DIR / "umap10.joblib"); joblib.dump(reducer2, MODEL_DIR / "umap2.joblib")
    joblib.dump(embeddings, MODEL_DIR / "training_embeddings.joblib"); joblib.dump([c["candidate_id"] for c in candidates], MODEL_DIR / "training_ids.joblib")
    joblib.dump(labels, MODEL_DIR / "cluster_labels.joblib"); joblib.dump({"centroids": centroids, "radii": radii}, MODEL_DIR / "cluster_geometry.joblib")
    print("\nSanity table")
    print("cluster | name | size | sim | phrases | time | domain | C | seed mix")
    for c in sorted(clusters, key=lambda x: -x["coordination_score"]):
        print(f"{c['cluster']:>7} | {c['display_name'][:28]:28} | {c['size']:>4} | {c['mean_similarity']:.2f} | {c['shared_phrase_ratio']:.2f} | {c['submission_concentration']:.2f} | {c['shared_portfolio_domain_ratio']:.2f} | {c['coordination_score']:>3} | {c['seed_mix']}")


if __name__ == "__main__":
    main()
