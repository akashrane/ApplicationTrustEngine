from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from shared.schema import NEW_ACCOUNT_DAYS

ROOT = Path(__file__).parent
LANGUAGE_MAP = {
    "python": {"python"}, "sql": {"sql", "plpgsql"}, "r": {"r"}, "java": {"java"},
    "javascript": {"javascript", "typescript"}, "typescript": {"typescript"}, "react": {"javascript", "typescript"},
    "node.js": {"javascript", "typescript"}, "go": {"go"}, "c++": {"c++"},
    "machine learning": {"python", "r"}, "pytorch": {"python"}, "tensorflow": {"python"},
    "scikit-learn": {"python"}, "pandas": {"python"}, "spark": {"scala", "python", "spark"},
    "airflow": {"python", "airflow"}, "aws": {"hcl", "python", "aws"},
}


def parse_date(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def earliest_start(rec: dict) -> datetime:
    years = [int(x["start"][:4]) for x in rec.get("experience", []) if x.get("start", "")[:4].isdigit()]
    return datetime(min(years) if years else parse_date(rec["submitted_at"]).year, 1, 1, tzinfo=timezone.utc)


def fetch_github(username: str, token: str = "") -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    user = requests.get(f"https://api.github.com/users/{username}", headers=headers, timeout=10)
    user.raise_for_status()
    repos_response = requests.get(f"https://api.github.com/users/{username}/repos?per_page=100&sort=created", headers=headers, timeout=10)
    repos_response.raise_for_status()
    repos = repos_response.json()
    return {
        "account_created": user.json()["created_at"],
        "public_repos": [{"name": r["name"], "language": r.get("language") or "", "created_at": r["created_at"], "pushed_at": r["pushed_at"]} for r in repos],
    }


def analyze_candidate(rec: dict, evidence: dict | None) -> dict:
    submitted = parse_date(rec["submitted_at"])
    start = earliest_start(rec)
    years = max(0, submitted.year - start.year)
    claims, timeline = [], [{"event": "Claimed experience begins", "date": start.date().isoformat(), "kind": "claim"}]
    contradiction = False
    account_age_days = None
    if evidence:
        created = parse_date(evidence["account_created"])
        account_age_days = (submitted - created).days
        contradiction = account_age_days < NEW_ACCOUNT_DAYS and years >= 3
        claims.append({
            "claim": f"{years}+ years of experience", "status": "contradicted" if contradiction else "supported",
            "evidence": (f"GitHub account appeared {account_age_days} days before submission despite a {years}-year timeline" if contradiction else f"GitHub history begins {created.date()}, consistent with the claimed timeline"),
        })
        claims.append({"claim": f"GitHub ownership: {rec['github_username']}", "status": "supported", "evidence": f"Cached account metadata exists from {created.date()}"})
        timeline.append({"event": "GitHub account created", "date": created.date().isoformat(), "kind": "account"})
        repos = evidence.get("public_repos", [])
        cutoff = submitted - timedelta(days=183)
        languages = {}
        for repo in repos:
            language = (repo.get("language") or "").casefold()
            created_at = parse_date(repo["created_at"])
            if created_at <= cutoff:
                languages.setdefault(language, []).append(repo)
            timeline.append({"event": repo["name"], "date": created_at.date().isoformat(), "kind": "repo", "language": repo.get("language", "")})
        for skill in rec.get("skills", []):
            matches = LANGUAGE_MAP.get(skill.casefold(), {skill.casefold()})
            supporting = [repo for language in matches for repo in languages.get(language, [])]
            if supporting:
                oldest = min(supporting, key=lambda r: r["created_at"])
                claims.append({"claim": f"Skill: {skill}", "status": "supported", "evidence": f"Repository “{oldest['name']}” predates submission by at least 6 months"})
            else:
                claims.append({"claim": f"Skill: {skill}", "status": "unverified", "evidence": "No matching repository old enough for automatic support"})
    else:
        claims.append({"claim": f"{years}+ years of experience", "status": "unverified", "evidence": "No linked GitHub history; absence is not negative evidence"})
        claims.append({"claim": "GitHub link", "status": "unverified", "evidence": "No GitHub username provided"})
        for skill in rec.get("skills", []):
            claims.append({"claim": f"Skill: {skill}", "status": "unverified", "evidence": "No linked GitHub history; absence is not negative evidence"})
    degree = rec.get("education", {})
    claims.append({"claim": f"Degree: {degree.get('degree', 'Not listed')}", "status": "unverified", "evidence": "Education verification was not requested for this prototype"})
    supported = sum(c["status"] == "supported" for c in claims)
    contradicted = sum(c["status"] == "contradicted" for c in claims)
    p_score = max(0, round(100 * supported / max(1, len(claims)) - 25 * contradicted))
    return {
        "candidate_id": rec["candidate_id"], "claims": claims, "P": p_score,
        "contradiction_count": contradicted, "github_account_age_days": account_age_days,
        "github_timeline": sorted(timeline, key=lambda x: x["date"]),
    }


def main(live: bool = False) -> None:
    load_dotenv(ROOT / ".env")
    records = json.loads((ROOT / "data/raw/resumes.json").read_text())
    cache_path = ROOT / "data/raw/github_cache.json"
    cache = json.loads(cache_path.read_text())
    if live:
        token = os.getenv("GITHUB_TOKEN", "")
        for rec in records:
            username = rec.get("github_username")
            if not username:
                continue
            try:
                cache[username] = fetch_github(username, token)
                print(f"Fetched @{username}")
            except requests.RequestException as exc:
                print(f"Keeping cached @{username}: {exc}")
        cache_path.write_text(json.dumps(cache, indent=2))
    output = [analyze_candidate(rec, cache.get(rec.get("github_username", ""))) for rec in records]
    (ROOT / "data/processed/provenance.json").write_text(json.dumps(output, indent=2))
    print(f"Provenance: {sum(x['contradiction_count'] for x in output)} contradictions; {sum(x['P'] == 0 for x in output)} records with no automatically supported share")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Refresh cached GitHub metadata")
    args = parser.parse_args()
    main(args.live)
