from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from datasets_manager import DatasetConnector
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


def evaluate_dataset(
    source: str = "all",
    limit: int = 20,
    job_spec_path: Path | None = None,
    force_llm: bool = False,
) -> dict:
    start_time = time.time()
    job_spec = json.loads(Path(job_spec_path).read_text()) if job_spec_path and Path(job_spec_path).exists() else JOB_SPEC

    parser = HybridResumeParser()
    connector = DatasetConnector(parser=parser)

    print(f"\n=======================================================")
    print(f" TESTING DATASET: [{source.upper()}] (Limit: {limit})")
    print(f" Target Job: {job_spec.get('title', 'Data Scientist')}")
    print(f"=======================================================")

    candidates = connector.load_dataset(source=source, limit=limit)
    if not candidates:
        print("No candidates retrieved from dataset.")
        return {}

    scored_candidates = []
    github_cache = json.loads((ROOT / "data/raw/github_cache.json").read_text()) if (ROOT / "data/raw/github_cache.json").exists() else {}
    scored_baseline = json.loads((ROOT / "data/processed/scored.json").read_text()) if (ROOT / "data/processed/scored.json").exists() else {"anomaly_baseline": [0.2, 0.4, 0.6]}

    local_count = 0
    llm_count = 0
    names_found = 0
    emails_found = 0
    githubs_found = 0
    experience_found = 0
    education_found = 0
    total_skills = 0

    quadrants = {"A": 0, "B": 0, "C": 0, "D": 0}
    actions = {}

    for cand in candidates:
        tier = cand.get("parser_tier", "local")
        if tier == "llm":
            llm_count += 1
        else:
            local_count += 1

        if cand.get("name") and cand["name"] != "Candidate":
            names_found += 1
        if cand.get("email"):
            emails_found += 1
        if cand.get("github_username"):
            githubs_found += 1
        if cand.get("experience") and len(cand["experience"]) > 0:
            experience_found += 1
        if cand.get("education") and (cand["education"].get("school") != "Not listed" or cand["education"].get("degree") != "Not listed"):
            education_found += 1
        total_skills += len(cand.get("skills", []))

        # Provenance Analysis
        gh_user = cand.get("github_username", "")
        evidence = github_cache.get(gh_user)
        prov = analyze_candidate(cand, evidence)

        # Qualification Scoring
        q_score, requirements = qualification_score(cand.get("text", ""), job_spec)

        # Anomaly & Trust
        mostly_unverified = sum(c["status"] == "unverified" for c in prov["claims"]) >= len(prov["claims"]) * 0.75
        raw_a = anomaly_raw(0.45, 0, prov["contradiction_count"], prov["github_account_age_days"], prov["P"], mostly_unverified)
        anomaly = percentile(raw_a, scored_baseline.get("anomaly_baseline", [0.3]))
        coordination = 0  # Single/batch external candidate baseline

        confidence, trust_score = trust_confidence(prov["P"], anomaly, coordination, prov["claims"])
        quad = quadrant(q_score, trust_score, confidence)
        quadrants[quad] = quadrants.get(quad, 0) + 1

        integrity = {"hidden_spans": [], "injection_detected": False, "security_events": []}
        action = verification_action(prov, integrity, coordination)
        actions[action] = actions.get(action, 0) + 1

        scored_candidates.append({
            "candidate_id": cand["candidate_id"],
            "name": cand["name"],
            "dataset_source": cand.get("dataset_source", source),
            "dataset_category": cand.get("dataset_category", "N/A"),
            "Q": q_score,
            "P": prov["P"],
            "A": anomaly,
            "C": coordination,
            "trust_score": trust_score,
            "trust_confidence": confidence,
            "quadrant": quad,
            "verification_action": action,
            "skills": cand.get("skills", []),
            "parser_tier": tier,
            "parser_confidence": cand.get("parser_confidence", 0.0),
        })

    elapsed = time.time() - start_time
    total = len(candidates)

    report = {
        "source": source,
        "total_evaluated": total,
        "elapsed_seconds": round(elapsed, 2),
        "avg_time_per_resume_sec": round(elapsed / max(1, total), 3),
        "parser_metrics": {
            "local_tier_pct": round(100 * local_count / total, 1),
            "llm_tier_pct": round(100 * llm_count / total, 1),
            "name_yield_pct": round(100 * names_found / total, 1),
            "email_yield_pct": round(100 * emails_found / total, 1),
            "github_yield_pct": round(100 * githubs_found / total, 1),
            "experience_yield_pct": round(100 * experience_found / total, 1),
            "education_yield_pct": round(100 * education_found / total, 1),
            "avg_skills_per_resume": round(total_skills / total, 1),
        },
        "quadrant_distribution": quadrants,
        "top_verification_actions": actions,
        "candidates": scored_candidates,
    }

    print("\nEVALUATION BENCHMARK RESULTS:")
    print(f"Total Candidates: {total} | Time: {elapsed:.2f}s ({report['avg_time_per_resume_sec']}s/resume)")
    print(f"Parser Tier: Local={report['parser_metrics']['local_tier_pct']}% | LLM Fallback={report['parser_metrics']['llm_tier_pct']}%")
    print(f"Extraction Yield: Name={report['parser_metrics']['name_yield_pct']}% | Email={report['parser_metrics']['email_yield_pct']}% | GitHub={report['parser_metrics']['github_yield_pct']}% | Exp={report['parser_metrics']['experience_yield_pct']}% | Edu={report['parser_metrics']['education_yield_pct']}%")
    print(f"Quadrant Distribution: Quadrant A={quadrants['A']} | Quadrant B={quadrants['B']} | Quadrant C={quadrants['C']} | Quadrant D={quadrants['D']}")
    print(f"Top Verification Actions:")
    for act, count in sorted(actions.items(), key=lambda x: x[1], reverse=True)[:4]:
        print(f"  - {act}: {count} candidate(s)")

    print("\nTop Scored Candidates:")
    for c in sorted(scored_candidates, key=lambda x: x["Q"], reverse=True)[:5]:
        print(f"  [{c['quadrant']}] {c['name']} (Q={c['Q']}, P={c['P']}, A={c['A']}) -> {c['verification_action']}")

    return report


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Test and benchmark external resume datasets with ApplicationTrust")
    cli_parser.add_argument("--source", default="all", choices=["huggingface", "resumeatlas", "kaggle", "innovatiana", "all"], help="Dataset source")
    cli_parser.add_argument("--limit", type=int, default=20, help="Number of records to test")
    cli_parser.add_argument("--job", type=str, default="data/job.json", help="Path to job.json")
    cli_parser.add_argument("--output", type=str, default="data/processed/dataset_benchmark.json", help="Path to save output report")
    args = cli_parser.parse_args()

    results = evaluate_dataset(source=args.source, limit=args.limit, job_spec_path=Path(args.job) if args.job else None)
    if results and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved benchmark results to: {out_path}")
