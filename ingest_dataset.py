from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from datasets_manager import DatasetConnector
from live import embed_one
from parser import HybridResumeParser
from provenance import analyze_candidate
from score import (
    COORDINATION_FLAG_THRESHOLD,
    JOB_SPEC,
    Q_THRESHOLD,
    anomaly_raw,
    percentile,
    qualification_score,
    quadrant,
    trust_confidence,
    verification_action,
)

ROOT = Path(__file__).parent
MODELS = ROOT / "data/models"


def ingest_dataset(
    source: str = "all",
    limit: int = 50,
    job_spec_path: Optional[Path] = None,
    output_path: Path = ROOT / "data/processed/external_scored.json",
) -> dict:
    job = json.loads(Path(job_spec_path).read_text()) if job_spec_path and Path(job_spec_path).exists() else JOB_SPEC

    print(f"\n=======================================================")
    print(f" INGESTING DATASET: [{source.upper()}] (Limit: {limit})")
    print(f" Target Job: {job.get('title', 'Data Scientist')}")
    print(f"=======================================================")

    parser = HybridResumeParser()
    connector = DatasetConnector(parser=parser)
    candidates = connector.load_dataset(source=source, limit=limit)

    if not candidates:
        print("No candidates retrieved.")
        return {}

    github_cache = json.loads((ROOT / "data/raw/github_cache.json").read_text()) if (ROOT / "data/raw/github_cache.json").exists() else {}
    scored_baseline = json.loads((ROOT / "data/processed/scored.json").read_text()) if (ROOT / "data/processed/scored.json").exists() else {"anomaly_baseline": [0.3]}

    training_embeddings = joblib.load(MODELS / "training_embeddings.joblib") if (MODELS / "training_embeddings.joblib").exists() else None
    reducer2 = joblib.load(MODELS / "umap2.joblib") if (MODELS / "umap2.joblib").exists() else None
    reducer10 = joblib.load(MODELS / "umap10.joblib") if (MODELS / "umap10.joblib").exists() else None
    geometry = joblib.load(MODELS / "cluster_geometry.joblib") if (MODELS / "cluster_geometry.joblib").exists() else None
    population_data = json.loads((ROOT / "data/processed/population.json").read_text()) if (ROOT / "data/processed/population.json").exists() else {"clusters": []}

    scored_candidates = []
    for cand in candidates:
        text = cand.get("text", "")
        # Embed candidate
        embedding = embed_one(text) if (MODELS / "embedder.joblib").exists() else np.zeros((1, 10))
        
        # 2D projection
        if reducer2 is not None:
            point2 = reducer2.transform(embedding)[0]
            point10 = reducer10.transform(embedding)[0] if reducer10 else None
            x, y = float(point2[0]), float(point2[1])
        else:
            x, y = 0.0, 0.0
            point10 = None

        # Nearest neighbor & cluster matching
        nn_max_sim = 0.5
        cluster_id = -1
        cluster_name = "External Applicant"
        coordination = 0

        if training_embeddings is not None and len(training_embeddings) > 0:
            sims = cosine_similarity(embedding, training_embeddings)[0]
            nn_max_sim = float(sims.max())

        if geometry and point10 is not None:
            choices = [(label, float(np.linalg.norm(point10 - center))) for label, center in geometry["centroids"].items()]
            label, distance = min(choices, key=lambda item: item[1]) if choices else (-1, float("inf"))
            if label != -1 and distance <= geometry["radii"].get(label, 0):
                cluster_id = label
                matched_c = next((c for c in population_data.get("clusters", []) if c["cluster"] == label), None)
                if matched_c:
                    cluster_name = matched_c["display_name"]
                    coordination = matched_c["coordination_score"]

        # Provenance
        gh_user = cand.get("github_username", "")
        evidence = github_cache.get(gh_user)
        prov = analyze_candidate(cand, evidence)

        # Qualification
        q_score, requirements = qualification_score(text, job)

        # Anomaly & Trust
        mostly_unverified = sum(c["status"] == "unverified" for c in prov["claims"]) >= len(prov["claims"]) * 0.75
        raw_a = anomaly_raw(nn_max_sim, 0, prov["contradiction_count"], prov["github_account_age_days"], prov["P"], mostly_unverified)
        anomaly = percentile(raw_a, scored_baseline.get("anomaly_baseline", [0.3]))

        confidence, trust_score = trust_confidence(prov["P"], anomaly, coordination, prov["claims"])
        quad = quadrant(q_score, trust_score, confidence)

        integrity = {"hidden_spans": [], "injection_detected": False, "security_events": []}
        action = verification_action(prov, integrity, coordination)

        scored_candidates.append({
            **cand,
            "Q": q_score,
            "P": prov["P"],
            "A": anomaly,
            "C": coordination,
            "x": x,
            "y": y,
            "cluster": cluster_id,
            "cluster_name": cluster_name,
            "nn_max_similarity": round(nn_max_sim, 3),
            "trust_confidence": confidence,
            "trust_score": trust_score,
            "quadrant": quad,
            "verification_action": action,
            "requirements": requirements,
            "claims": prov["claims"],
            "github_timeline": prov.get("github_timeline", []),
            "github_account_age_days": prov.get("github_account_age_days"),
            "contradiction_count": prov.get("contradiction_count", 0),
            "hidden_spans": [],
            "injection_detected": False,
            "security_events": [],
            "duplicate_of": None,
            "status": "active",
        })

    # Priority ranking
    top = sorted([c for c in scored_candidates if c["quadrant"] == "A"], key=lambda c: (-c["Q"], -c["P"], c["A"], c["candidate_id"]))
    top_ids = [c["candidate_id"] for c in top]
    for c in scored_candidates:
        c["priority_rank"] = top_ids.index(c["candidate_id"]) + 1 if c["candidate_id"] in top_ids else None

    quad_counts = {"A": sum(c["quadrant"] == "A" for c in scored_candidates),
                   "B": sum(c["quadrant"] == "B" for c in scored_candidates),
                   "C": sum(c["quadrant"] == "C" for c in scored_candidates),
                   "D": sum(c["quadrant"] == "D" for c in scored_candidates)}

    output = {
        "dataset_source": source,
        "job": job,
        "total_ingested": len(scored_candidates),
        "quadrant_counts": quad_counts,
        "candidates": scored_candidates,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Successfully ingested & scored {len(scored_candidates)} candidates.")
    print(f"Quadrant Distribution: A={quad_counts['A']} | B={quad_counts['B']} | C={quad_counts['C']} | D={quad_counts['D']}")
    print(f"Saved to: {output_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and score external resume datasets")
    parser.add_argument("--source", default="all", choices=["huggingface", "resumeatlas", "kaggle", "innovatiana", "all"], help="Dataset source")
    parser.add_argument("--limit", type=int, default=50, help="Number of records to ingest")
    parser.add_argument("--job", type=str, default="data/job.json", help="Path to job.json")
    parser.add_argument("--output", type=str, default="data/processed/external_scored.json", help="Output path")
    args = parser.parse_args()

    ingest_dataset(source=args.source, limit=args.limit, job_spec_path=Path(args.job) if args.job else None, output_path=Path(args.output))
