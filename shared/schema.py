from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

RANDOM_STATE = 42
Q_THRESHOLD = 70
COORDINATION_FLAG_THRESHOLD = 55
HIGH_COORDINATION_THRESHOLD = 70
NEW_ACCOUNT_DAYS = 30

JOB_SPEC = {
    "title": "Data Scientist",
    "must_have": ["Python", "SQL", "3+ years experience"],
    "preferred": ["Spark", "AWS", "Machine Learning", "Airflow"],
}

SKILL_VOCABULARY = [
    "Python", "SQL", "R", "Java", "JavaScript", "TypeScript", "React",
    "Node.js", "Go", "C++", "Spark", "AWS", "GCP", "Azure", "Airflow",
    "Machine Learning", "TensorFlow", "PyTorch", "scikit-learn", "Pandas",
    "Tableau", "Power BI", "dbt", "Snowflake", "Docker", "Kubernetes",
    "PostgreSQL", "MongoDB", "Kafka", "Git", "Statistics", "Excel",
]

ClaimStatus = Literal["supported", "unverified", "contradicted"]


class Claim(TypedDict):
    claim: str
    status: ClaimStatus
    evidence: str


@dataclass
class Candidate:
    candidate_id: str
    name: str
    email: str
    phone: str
    submitted_at: str
    github_username: str = ""
    portfolio_url: str = ""
    education: dict = field(default_factory=dict)
    experience: list[dict] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    summary: str = ""
    text: str = ""
    pdf_path: str | None = None
    seed_tag: str = "normal"

