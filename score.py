from __future__ import annotations

import json
import re
from pathlib import Path

from shared.schema import COORDINATION_FLAG_THRESHOLD, JOB_SPEC, Q_THRESHOLD

ROOT = Path(__file__).parent


def years_in_text(text: str) -> int:
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    return max(years, default=0) - min(years, default=0) if len(years) >= 2 else 0


def qualification_score(text: str, job: dict = JOB_SPEC) -> tuple[int, list[dict]]:
    lowered = text.casefold()
    checks = []
    for requirement in job["must_have"]:
        if requirement.casefold().startswith("3+"):
            met = years_in_text(text) >= 3 or bool(re.search(r"\b(?:3|4|5|6|7|8|9|10)\+? years\b", lowered))
        else:
            met = requirement.casefold() in lowered
        checks.append({"requirement": requirement, "type": "must-have", "met": met})
    for requirement in job["preferred"]:
        checks.append({"requirement": requirement, "type": "preferred", "met": requirement.casefold() in lowered})
    must = sum(c["met"] for c in checks if c["type"] == "must-have") / len(job["must_have"])
    preferred = sum(c["met"] for c in checks if c["type"] == "preferred") / len(job["preferred"])
    return round(70 * must + 30 * preferred), checks


def anomaly_raw(nn_similarity: float, hidden_count: int, contradiction_count: int, account_age_days: int | None, p_score: int, mostly_unverified: bool = False) -> float:
    youth = 1.0 if account_age_days is not None and account_age_days < 30 else 0.0
    low_p = 0.0 if mostly_unverified and contradiction_count == 0 else 1 - p_score / 100
    return 0.35 * nn_similarity + 0.20 * min(1, hidden_count / 2) + 0.20 * min(1, contradiction_count) + 0.15 * youth + 0.10 * low_p


def percentile(value: float, baseline: list[float]) -> int:
    return round(100 * sum(x <= value for x in baseline) / max(1, len(baseline)))


def trust_confidence(p_score: int, anomaly: int, coordination: int, claims: list[dict]) -> tuple[str, int]:
    resolved = sum(c["status"] != "unverified" for c in claims)
    contradicted = sum(c["status"] == "contradicted" for c in claims)
    if resolved <= 1 and not contradicted:
        return "UNKNOWN", 50
    trust = round(0.55 * p_score + 0.25 * (100 - anomaly) + 0.20 * (100 - coordination))
    if contradicted or anomaly >= 75 or coordination >= 70:
        return "LOW", trust
    if p_score >= 60 and anomaly < 60 and coordination < COORDINATION_FLAG_THRESHOLD:
        return "HIGH", trust
    return "MEDIUM", trust


def quadrant(q_score: int, trust_score: int, confidence: str) -> str:
    high_q = q_score >= Q_THRESHOLD
    high_trust = trust_score >= 60 and confidence in {"HIGH", "MEDIUM"}
    return ("A" if high_trust else "B") if high_q else ("D" if high_trust else "C")


def verification_action(prov: dict, integrity: dict, coordination: int) -> str:
    claims = prov.get("claims", [])
    github_issue = prov.get("github_account_age_days") is not None and prov["github_account_age_days"] < 30
    github_issue |= any(c["status"] == "contradicted" and "GitHub" in c["evidence"] for c in claims)
    employment_issue = any(c["status"] == "contradicted" and "experience" in c["claim"].casefold() for c in claims)
    high_severity = sum(e.get("severity") == "high" for e in integrity.get("security_events", [])) + prov.get("contradiction_count", 0)
    if high_severity >= 3:
        return "Identity verification (Persona/Checkr)"
    if github_issue:
        return "GitHub OAuth ownership check"
    if employment_issue:
        return "Request employment proof"
    if coordination >= 70:
        return "Recruiter cluster review"
    if integrity.get("injection_detected"):
        return "Document security review"
    return "No action"


def main() -> None:
    resumes = json.loads((ROOT / "data/raw/resumes.json").read_text())
    integrity = {x["candidate_id"]: x for x in json.loads((ROOT / "data/processed/integrity.json").read_text())}
    population_data = json.loads((ROOT / "data/processed/population.json").read_text())
    population = {x["candidate_id"]: x for x in population_data["candidates"]}
    provenance = {x["candidate_id"]: x for x in json.loads((ROOT / "data/processed/provenance.json").read_text())}
    job = json.loads((ROOT / "data/job.json").read_text())
    raw_values, prelim = [], []
    for rec in resumes:
        cid, check, prov = rec["candidate_id"], integrity[rec["candidate_id"]], provenance[rec["candidate_id"]]
        point = population.get(cid, {"x": None, "y": None, "cluster": -1, "cluster_name": "Collapsed duplicate", "nn_max_similarity": 0, "C": 0})
        q_score, requirements = qualification_score(check["sanitized_text"], job)
        mostly_unverified = sum(c["status"] == "unverified" for c in prov["claims"]) >= len(prov["claims"]) * 0.75
        raw = anomaly_raw(point["nn_max_similarity"], len(check["hidden_spans"]), prov["contradiction_count"], prov["github_account_age_days"], prov["P"], mostly_unverified)
        raw_values.append(raw)
        prelim.append((rec, check, prov, point, q_score, requirements, raw))
    candidates = []
    for rec, check, prov, point, q_score, requirements, raw in prelim:
        anomaly = 0 if check["duplicate_of"] else percentile(raw, [v for i, v in enumerate(raw_values) if not integrity[resumes[i]["candidate_id"]]["duplicate_of"]])
        confidence, trust_score = trust_confidence(prov["P"], anomaly, point["C"], prov["claims"])
        quad = "collapsed" if check["duplicate_of"] else quadrant(q_score, trust_score, confidence)
        action = "No action" if check["duplicate_of"] else verification_action(prov, check, point["C"])
        candidates.append({
            **rec, "text": check["sanitized_text"], "Q": q_score, "P": prov["P"], "A": anomaly, "C": point["C"],
            "requirements": requirements, "claims": prov["claims"], "github_timeline": prov["github_timeline"],
            "github_account_age_days": prov["github_account_age_days"], "contradiction_count": prov["contradiction_count"],
            "hidden_spans": check["hidden_spans"], "injection_detected": check["injection_detected"], "security_events": check["security_events"],
            "duplicate_of": check["duplicate_of"], "x": point["x"], "y": point["y"], "cluster": point["cluster"],
            "cluster_name": point["cluster_name"], "nn_max_similarity": point["nn_max_similarity"],
            "trust_confidence": confidence, "trust_score": trust_score, "quadrant": quad, "verification_action": action,
            "status": "collapsed" if check["duplicate_of"] else ("security_event" if check["injection_detected"] else "active"),
        })
    top = sorted([c for c in candidates if c["quadrant"] == "A"], key=lambda c: (-c["Q"], -c["P"], c["A"], c["candidate_id"]))[:100]
    top_ids = [c["candidate_id"] for c in top]
    for c in candidates:
        c["priority_rank"] = top_ids.index(c["candidate_id"]) + 1 if c["candidate_id"] in top_ids else None
    nonduplicates = [c for c in candidates if not c["duplicate_of"]]
    funnel = [
        {"stage": "Applications received", "value": len(candidates)},
        {"stage": "Unique applications", "value": len(nonduplicates)},
        {"stage": "No security event", "value": sum(not c["injection_detected"] for c in nonduplicates)},
        {"stage": "Meets all must-haves", "value": sum(all(r["met"] for r in c["requirements"] if r["type"] == "must-have") for c in nonduplicates)},
        {"stage": "High qualification", "value": sum(c["Q"] >= Q_THRESHOLD for c in nonduplicates)},
        {"stage": "Top 100", "value": len(top)},
    ]
    output = {
        "generated_at": max(c["submitted_at"] for c in resumes), "job": job, "embedding_method": population_data["embedding_method"],
        "candidates": candidates, "top_100": top_ids, "funnel": funnel,
        "counts": {"duplicates_collapsed": sum(bool(c["duplicate_of"]) for c in candidates), "security_events": sum(c["injection_detected"] for c in candidates),
                   "step_up": sum(c["quadrant"] == "B" for c in candidates), "cluster_review": sum(c["C"] >= 70 for c in candidates)},
        "anomaly_baseline": raw_values,
    }
    (ROOT / "data/processed/scored.json").write_text(json.dumps(output, indent=2))
    by_id = {c["candidate_id"]: c for c in candidates}
    assert by_id["C0007"]["quadrant"] == "B" and by_id["C0007"]["injection_detected"] and by_id["C0007"]["contradiction_count"] > 0
    assert by_id["C0012"]["quadrant"] == "A" and by_id["C0012"]["priority_rank"] and by_id["C0012"]["priority_rank"] <= 10
    print(f"Scored {len(candidates)} candidates: Top 100={len(top)}, step-up={output['counts']['step_up']}, security events={output['counts']['security_events']}")
    print(f"C0007: Q={by_id['C0007']['Q']} P={by_id['C0007']['P']} A={by_id['C0007']['A']} C={by_id['C0007']['C']} quadrant={by_id['C0007']['quadrant']} action={by_id['C0007']['verification_action']}")
    print(f"C0012: Q={by_id['C0012']['Q']} P={by_id['C0012']['P']} A={by_id['C0012']['A']} C={by_id['C0012']['C']} quadrant={by_id['C0012']['quadrant']} rank={by_id['C0012']['priority_rank']}")


if __name__ == "__main__":
    main()
