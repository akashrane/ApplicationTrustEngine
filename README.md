# Applicant Trust Intelligence
Recruiter decision-support prototype that analyzes applicants individually and as a population.
It reports Qualification, Provenance, Anomaly, and Coordination separately; no score auto-rejects a candidate.
Python 3.10+ is required. Copy `.env.example` to `.env` only for optional OpenAI/GitHub live features.
Run `make install`, then `make all`, then `make app`.
The offline pipeline uses deterministic synthetic data and falls back to TF-IDF/SVD when transformer embeddings are unavailable.
The requested cohorts sum to 530, so this 500-record build uses 360 normal records while preserving every anomaly cohort size.
Demo Jordan Reyes (`C0007`): Landscape → Cluster Inspector → security event → recent GitHub → Quadrant B step-up.
Demo Priya Nair (`C0012`): Candidate Profile → long-lived GitHub evidence → Quadrant A → Priority Queue top 10.
Signals prioritize human review; the interface uses evidence-aware, NYC AEDT-conscious language.
