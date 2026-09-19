from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data/processed"

QUADRANT_NAMES = {
    "A": "Quadrant A · Priority Fast-Track",
    "B": "Quadrant B · Step-Up Verification",
    "C": "Quadrant C · Low Match / Screened Out",
    "D": "Quadrant D · Talent Pool / Alternative Roles",
}

QUADRANT_DESCRIPTIONS = {
    "A": "High Qualification Fit & Verified Evidence. Ready for immediate recruiter interview.",
    "B": "High Qualification Fit with Unverified or Flagged Trust. Requires targeted verification before decision. Never auto-rejected.",
    "C": "Does not meet core job requirements and has low trust signals. Screened out of active consideration.",
    "D": "Authentic candidate with high trust evidence who does not fit this specific role. Retain in talent pool for future openings.",
}

st.set_page_config(
    page_title="Applicant Trust Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --ink-primary: #0f172a;
    --ink-secondary: #475569;
    --ink-muted: #64748b;
    --brand-teal: #0f766e;
    --brand-blue: #1d4ed8;
    --brand-amber: #b45309;
    --brand-red: #b91c1c;
    --surface-bg: #f8fafc;
    --card-bg: #ffffff;
    --border-color: #e2e8f0;
}

.stApp {
    background: var(--surface-bg);
    color: var(--ink-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

[data-testid="stSidebar"] {
    background: #0f172a;
}
[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

.hero-banner {
    padding: 1.5rem 1.8rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f766e 100%);
    color: #ffffff;
    margin-bottom: 1.25rem;
    border: 1px solid #334155;
}
.hero-banner h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.hero-banner p {
    margin: 0.4rem 0 0;
    color: #cbd5e1;
    font-size: 0.95rem;
}

.content-card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.status-A, .status-HIGH { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.status-B, .status-MEDIUM { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.status-C, .status-LOW { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
.status-D, .status-UNKNOWN { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }

mark {
    background: #fef08a;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    font-weight: 600;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--border-color);
    padding: 1rem;
    border-radius: 10px;
}
[data-testid="stMetric"] * {
    color: var(--ink-primary) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data() -> tuple[dict, dict]:
    return json.loads((PROCESSED / "scored.json").read_text()), json.loads((PROCESSED / "population.json").read_text())


try:
    scored, population = load_data()
except FileNotFoundError:
    st.error("Processed data is missing. Please run `make all` in your terminal first.")
    st.stop()

candidates = scored["candidates"]
by_id = {c["candidate_id"]: c for c in candidates}
active = [c for c in candidates if not c.get("duplicate_of")]
clusters = population.get("clusters", [])
job_spec = scored.get("job", {})


def heading(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero-banner"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def terms_glossary_expander() -> None:
    with st.expander("Reference Guide: Scores, Quadrants & Screening Stages", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Evaluation Scores (Range: 0 to 100)**
            * **Q · Qualification Fit**: Match rate against job requirements (must-have skills, preferred technologies, and total years of relevant experience). Higher is better.
            * **P · Provenance (Verified History)**: Percentage of claimed timeline and skills backed by public evidence (GitHub activity, repository creation dates, account longevity). Higher is better.
            * **A · Anomaly Score**: Measures unusual characteristics like hidden text, prompt injection attempts, or timeline contradictions. Lower is clean; higher is anomalous.
            * **C · Coordination Score**: Measures population-level group collision (synchronized submission spikes, identical templates, shared portfolio websites). Lower is independent; higher is coordinated.
            """)
        with c2:
            st.markdown("""
            **Decision Quadrants**
            * **Quadrant A · Priority Fast-Track**: High Qualification Fit + High Trust Evidence. Top priority for immediate recruiter interviews.
            * **Quadrant B · Step-Up Verification**: High Qualification Fit + Unverified or Flagged Trust. Strong candidate on paper who requires a specific human check (e.g. GitHub check, work sample). Never auto-rejected.
            * **Quadrant C · Low Match / Screened Out**: Does not meet essential job requirements.
            * **Quadrant D · Talent Pool / Alternative Roles**: Authentic applicant with strong trust signals who is a mismatch for this specific job spec. Retain in talent pool.
            """)


def determine_funnel_stage(c: dict) -> dict:
    if c.get("duplicate_of"):
        return {
            "stage_num": 1,
            "stage_name": "Stage 1: Intake & Duplication",
            "status": "Collapsed Duplicate",
            "passed": False,
            "reason": f"Exact duplicate application of record {c['duplicate_of']}",
            "action": "Collapsed automatically (Zero recruiter time spent)",
            "badge_class": "status-C",
        }
    if c.get("injection_detected"):
        return {
            "stage_num": 1,
            "stage_name": "Stage 1: Document Integrity & Security",
            "status": "Security Alert: Prompt Injection",
            "passed": False,
            "reason": "Manipulation Detected: Hidden text or system prompt override instructions inside PDF",
            "action": "Document Security Audit",
            "badge_class": "status-B",
        }

    missing_must_haves = [r["requirement"] for r in c.get("requirements", []) if r.get("type") == "must-have" and not r.get("met")]
    if missing_must_haves:
        return {
            "stage_num": 2,
            "stage_name": "Stage 2: Must-Have Requirements",
            "status": "Missing Core Requirements",
            "passed": False,
            "reason": f"Missing required qualifications: {', '.join(missing_must_haves)}",
            "action": "Route to Quadrant C (Low Match) or Quadrant D (Talent Pool)",
            "badge_class": "status-C",
        }

    if c.get("C", 0) >= 70:
        return {
            "stage_num": 3,
            "stage_name": "Stage 3: Population Coordination",
            "status": "Template Syndicate Flag",
            "passed": False,
            "reason": f"Shared portfolio domain & synchronized phrasing with {c.get('cluster_name', 'Flagged Cluster')} (Coordination = {c.get('C')}%)",
            "action": "Recruiter Cluster Review (Quadrant B)",
            "badge_class": "status-B",
        }

    if c.get("contradiction_count", 0) > 0:
        age_days = c.get("github_account_age_days", 0)
        return {
            "stage_num": 4,
            "stage_name": "Stage 4: History & Provenance",
            "status": "Timeline Contradiction",
            "passed": False,
            "reason": f"GitHub account created {age_days} days ago despite claiming multi-year senior experience",
            "action": "GitHub OAuth Ownership Check (Quadrant B)",
            "badge_class": "status-B",
        }

    quad = c.get("quadrant", "C")
    if quad == "A":
        return {
            "stage_num": 5,
            "stage_name": "Stage 5: Final Routing",
            "status": "Fast-Track Priority",
            "passed": True,
            "reason": f"Meets all core requirements (Q={c.get('Q')}%) with verified history (P={c.get('P')}%)",
            "action": f"Priority Review (Rank #{c.get('priority_rank', 'Top')})",
            "badge_class": "status-A",
        }
    elif quad == "B":
        return {
            "stage_num": 5,
            "stage_name": "Stage 5: Final Routing",
            "status": "Step-Up Verification Required",
            "passed": False,
            "reason": f"Meets qualifications (Q={c.get('Q')}%), but requires human verification of claims",
            "action": c.get("verification_action", "Step-Up Verification"),
            "badge_class": "status-B",
        }
    elif quad == "D":
        return {
            "stage_num": 5,
            "stage_name": "Stage 5: Final Routing",
            "status": "Talent Pool / Alternative Roles",
            "passed": True,
            "reason": "Authentic candidate with high trust evidence, but lower qualification match for this specific opening",
            "action": "Retain in Talent Pool",
            "badge_class": "status-D",
        }
    else:
        return {
            "stage_num": 5,
            "stage_name": "Stage 5: Final Routing",
            "status": "Screened Out (Low Match)",
            "passed": False,
            "reason": f"Lower qualification match (Q={c.get('Q')}%)",
            "action": "Standard Low-Priority Review",
            "badge_class": "status-C",
        }


funnel_records = []
for c in candidates:
    info = determine_funnel_stage(c)
    funnel_records.append({**c, **info})


def gauge(value: int, label: str, color: str, inverse: bool = False) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label, "font": {"size": 13, "color": "#0f172a"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "bgcolor": "#e2e8f0",
            "steps": (
                [{"range": [0, 35], "color": "#dcfce7"}, {"range": [70, 100], "color": "#fee2e2"}]
                if inverse else
                [{"range": [0, 40], "color": "#fee2e2"}, {"range": [75, 100], "color": "#dcfce7"}]
            ),
        }
    ))
    fig.update_layout(height=170, margin=dict(l=15, r=15, t=35, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def score_row(candidate: dict) -> None:
    cols = st.columns(4)
    labels = [
        ("Q · Qualification Fit", "#0f766e", False),
        ("P · Provenance (Verified)", "#1d4ed8", False),
        ("A · Anomaly Signal", "#b45309", True),
        ("C · Group Coordination", "#7c3aed", True),
    ]
    for col, key, (lbl, color, inverse) in zip(cols, ["Q", "P", "A", "C"], labels):
        col.plotly_chart(gauge(candidate.get(key, 0), lbl, color, inverse), width="stretch", key=f"g-{candidate['candidate_id']}-{key}")


def landscape_figure(extra: dict | None = None) -> go.Figure:
    rows = [{**c, "cluster_color": c.get("cluster_name", "Natural applicants") if c.get("cluster", -1) != -1 else "Natural applicants"} for c in active]
    frame = pd.DataFrame(rows)
    fig = px.scatter(
        frame, x="x", y="y", color="cluster_color", hover_name="name",
        hover_data={"candidate_id": True, "Q": True, "P": True, "A": True, "C": True, "x": False, "y": False},
        color_discrete_sequence=px.colors.qualitative.Safe, opacity=0.75,
        title="2D Population Semantic Embedding Landscape"
    )
    fig.update_traces(marker={"size": 8})
    if extra and "x" in extra and "y" in extra:
        fig.add_trace(go.Scatter(
            x=[extra["x"]], y=[extra["y"]], mode="markers+text", name="Uploaded Resume",
            text=[extra.get("name", "Uploaded")], textposition="top center",
            marker={"symbol": "diamond", "size": 18, "color": "#dc2626", "line": {"width": 2, "color": "white"}}
        ))
    fig.update_layout(height=560, legend_title="Applicant Cohort / Cluster", xaxis_title="UMAP Dimension 1", yaxis_title="UMAP Dimension 2", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def quadrant_figure(rows: list[dict], extra: dict | None = None) -> go.Figure:
    frame = pd.DataFrame([c for c in rows if not c.get("duplicate_of")])
    frame["quadrant_label"] = frame["quadrant"].map(QUADRANT_NAMES)
    color_map = {
        QUADRANT_NAMES["A"]: "#0f766e",
        QUADRANT_NAMES["B"]: "#b45309",
        QUADRANT_NAMES["C"]: "#b91c1c",
        QUADRANT_NAMES["D"]: "#1d4ed8",
    }
    fig = px.scatter(
        frame, x="trust_score", y="Q", color="quadrant_label", hover_name="name",
        hover_data=["candidate_id", "P", "A", "C", "verification_action"],
        color_discrete_map=color_map, opacity=0.75,
        title="Trust Confidence Index vs Qualification Fit"
    )
    if extra and "trust_score" in extra and "Q" in extra:
        fig.add_trace(go.Scatter(
            x=[extra["trust_score"]], y=[extra["Q"]], mode="markers+text", name="Uploaded Resume",
            text=[extra.get("name", "Uploaded")], textposition="top center",
            marker={"symbol": "diamond", "size": 18, "color": "#0f172a", "line": {"width": 2, "color": "white"}}
        ))
    fig.add_vline(x=60, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=70, line_dash="dash", line_color="#94a3b8")
    for x, y, q_key in [(80, 95, "A"), (30, 95, "B"), (30, 20, "C"), (80, 20, "D")]:
        fig.add_annotation(
            x=x, y=y, text=QUADRANT_NAMES[q_key], showarrow=False,
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#cbd5e1", borderpad=5
        )
    fig.update_layout(height=580, xaxis_title="Trust Confidence Index (0 to 100)", yaxis_title="Qualification Fit Score (0 to 100)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def highlighted(text: str, phrases: list[str]) -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for phrase in sorted(phrases[:8], key=len, reverse=True):
        if phrase.strip():
            safe = re.sub(re.escape(phrase), lambda m: f"<mark>{m.group(0)}</mark>", safe, flags=re.I)
    return safe.replace("\n", "<br>")


def candidate_profile(candidate: dict, prefix: str = "profile") -> None:
    f_info = determine_funnel_stage(candidate)
    quad_key = candidate.get("quadrant", "C")
    quad_full_name = QUADRANT_NAMES.get(quad_key, f"Quadrant {quad_key}")
    
    st.markdown(f"""
    <div class="content-card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <h2 style="margin:0; font-size:1.5rem; color:#0f172a;">{candidate.get('name', 'Candidate')}</h2>
                <p style="margin:0.2rem 0; color:#64748b; font-size:0.9rem;">
                    Candidate ID: <b>{candidate.get('candidate_id', 'N/A')}</b> &nbsp;|&nbsp; 
                    Role: <b>{candidate.get('experience', [{'title':'Candidate'}])[0].get('title', 'Candidate')}</b> &nbsp;|&nbsp; 
                    Submitted: <b>{candidate.get('submitted_at', '')[:10]}</b> &nbsp;|&nbsp; 
                    Source: <span class="status-badge status-D">{candidate.get('seed_tag', 'Direct')}</span>
                </p>
            </div>
            <div>
                <span class="status-badge status-{quad_key}">{quad_full_name}</span> &nbsp;
                <span class="status-badge status-{candidate.get('trust_confidence', 'MEDIUM')}">{candidate.get('trust_confidence', 'MEDIUM')} TRUST CONFIDENCE</span>
            </div>
        </div>
        <div style="margin-top:0.8rem; padding:0.75rem 1rem; background:#f8fafc; border-radius:8px; border-left:4px solid {'#15803d' if f_info['passed'] else '#b45309'}; font-size:0.9rem;">
            <b>Verdict / Status:</b> {f_info['status']} — <i>{f_info['reason']}</i><br>
            <b>Recommended Action:</b> <code>{candidate.get('verification_action', 'Review')}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    score_row(candidate)

    tabs = st.tabs([
        "Job Fit & Skills Match",
        "Experience & Education History",
        "Verified Claims & Timeline",
        "Document Security Audit",
        "Similar Applicants in Pool",
        "Raw Extracted Text",
    ])

    with tabs[0]:
        st.subheader("Job Specification Requirements Match")
        req_rows = []
        for r in candidate.get("requirements", []):
            req_rows.append({
                "Match Status": "MET" if r["met"] else "MISSING",
                "Requirement": r["requirement"],
                "Priority Type": r["type"].upper(),
                "Evidence Note": "Identified in resume text" if r["met"] else "Not found in resume text",
            })
        st.dataframe(req_rows, hide_index=True, width="stretch")
        
        st.markdown(f"**Extracted Technical Skills ({len(candidate.get('skills', []))}):**")
        if candidate.get("skills"):
            skill_badges = " ".join([f"<span class='status-badge status-D' style='margin:2px;'>{s}</span>" for s in candidate["skills"]])
            st.markdown(skill_badges, unsafe_allow_html=True)
        else:
            st.caption("No specific skills isolated.")

    with tabs[1]:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Work History")
            exps = candidate.get("experience", [])
            if not exps:
                st.caption("No structured work experience parsed.")
            for exp in exps:
                st.markdown(f"**{exp.get('title', 'Role')}** @ **{exp.get('company', 'Company')}**")
                st.caption(f"Period: {exp.get('start', '')} to {exp.get('end', 'Present')}")
                for b in exp.get("bullets", []):
                    st.markdown(f"- {b}")
                st.divider()
        with c2:
            st.subheader("Education")
            edu = candidate.get("education", {})
            st.markdown(f"**Degree:** {edu.get('degree', 'Not listed')}")
            st.markdown(f"**Institution:** {edu.get('school', 'Not listed')}")
            st.markdown(f"**Graduation Year:** {edu.get('year', 'Not listed')}")

    with tabs[2]:
        st.subheader("Provenance Claims & Timeline Verification")
        st.dataframe(
            candidate.get("claims", []),
            hide_index=True,
            width="stretch",
            column_config={
                "status": st.column_config.TextColumn("Status"),
                "claim": st.column_config.TextColumn("Candidate Claim", width="medium"),
                "evidence": st.column_config.TextColumn("Verified Evidence", width="large"),
            }
        )
        
        timeline_data = candidate.get("github_timeline", [])
        if timeline_data:
            st.subheader("Historical Activity Timeline")
            t_df = pd.DataFrame(timeline_data)
            t_df["date"] = pd.to_datetime(t_df["date"])
            fig = px.scatter(
                t_df, x="date", y="kind", color="kind", hover_name="event",
                title="Account and Repository Timeline Events",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_traces(marker={"size": 11})
            fig.update_layout(height=260, yaxis_title="Activity Kind", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch", key=f"t-{prefix}-{candidate['candidate_id']}")
        else:
            st.info("No public GitHub timeline provided. Note: Absence of GitHub is treated as unverified, never as negative evidence.")

    with tabs[3]:
        st.subheader("Document Integrity & Anti-Tampering")
        sec_events = candidate.get("security_events", [])
        if not sec_events and not candidate.get("injection_detected"):
            st.success("Document is clean. No hidden text or prompt injection attempts detected.")
        else:
            st.error("Security Warning: Manipulation signals detected in document.")
            for ev in sec_events:
                st.warning(f"**{ev['type'].replace('_', ' ').title()}** ({ev.get('severity', 'medium').upper()} severity)\n\nReason: {ev.get('reason', '')}\n\nExtracted Content: `{ev.get('text', '')}`")
        st.caption("Qualification scores are calculated strictly from sanitized text to prevent prompt injection overrides.")

    with tabs[4]:
        st.subheader("Nearest Neighbor Candidates in Population")
        sims = candidate.get("similar_candidates", [])
        if sims:
            st.dataframe(sims, hide_index=True, width="stretch")
        else:
            st.caption("No nearest neighbor records cached for this candidate.")

    with tabs[5]:
        st.subheader("Extracted Plain Text")
        st.text_area("Plain Text Content", value=candidate.get("text", ""), height=320)


# =============================================================================
# NAVIGATION
# =============================================================================
pages = [
    "Overview & Executive Dashboard",
    "Hiring Funnel & Decision Audit",
    "Candidate Dossier & Profile",
    "Applicant Landscape (2D Embedding)",
    "Cluster & Template Inspector",
    "Quadrant Decision Matrix",
    "Priority Queue & Step-Up Routing",
    "External Datasets Benchmark",
    "Live Resume PDF Analyzer",
]

st.sidebar.markdown("## Applicant Trust Intelligence")
page = st.sidebar.radio("Navigation", pages)
st.sidebar.divider()
st.sidebar.caption(f"Embedding Engine: {scored.get('embedding_method', 'TF-IDF + SVD')}")
st.sidebar.caption(f"Target Role: {job_spec.get('title', 'Data Scientist')}")


# =============================================================================
# PAGE 1: OVERVIEW & EXECUTIVE DASHBOARD
# =============================================================================
if page == "Overview & Executive Dashboard":
    heading("Applicant Trust Intelligence", "Population-aware decision support for evidence-centered human review")
    terms_glossary_expander()

    metrics = st.columns(5)
    metrics[0].metric("Total Applicants", len(candidates))
    metrics[1].metric("Priority Queue (Top 100)", len(scored.get("top_100", [])))
    metrics[2].metric("Needs Step-Up Verify", scored["counts"]["step_up"])
    metrics[3].metric("Security Alerts", scored["counts"]["security_events"])
    metrics[4].metric("Coordinated Clusters", sum(c["coordination_score"] >= 55 for c in clusters))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Application Screening Funnel")
        funnel_df = pd.DataFrame(scored.get("funnel", []))
        st.plotly_chart(px.funnel(funnel_df, x="value", y="stage", color_discrete_sequence=["#0f766e"]), width="stretch")
    with right:
        st.subheader("Recruiter Workflow Guide")
        st.markdown("""
        1. **Priority Queue**: Review **Quadrant A · Priority Fast-Track** candidates who meet all requirements with supported evidence.
        2. **Step-Up Verification**: Review **Quadrant B · Step-Up Verification** candidates who have strong skills but require a specific verification step.
        3. **Decision Audit**: Use **Hiring Funnel & Decision Audit** to understand why candidates were flagged or screened out at each stage.
        4. **Benchmark**: Use **External Datasets Benchmark** to evaluate real-world public datasets.
        """)
        st.info("System Principle: Signals prioritize human review. No candidate is auto-rejected by algorithms.")

    st.divider()
    st.subheader("Active Document Security & Anti-Tampering Alerts")
    sec_flagged = [c for c in candidates if c.get("injection_detected") or c.get("security_events")]
    if sec_flagged:
        st.error(f"Security Alert: {len(sec_flagged)} applications flagged with active document tampering or adversarial prompt injection attempts.")
        sec_rows = []
        for c in sec_flagged:
            events = c.get("security_events", [])
            types = ", ".join(sorted(set(e.get("type", "manipulation").replace("_", " ").title() for e in events))) or "Prompt Injection"
            reasons = ", ".join(sorted(set(e.get("reason", "manipulation detected") for e in events)))
            payloads = " | ".join(f'"{e.get("text", "")}"' for e in events if e.get("text"))
            sec_rows.append({
                "Candidate ID": c["candidate_id"],
                "Applicant Name": c["name"],
                "Threat Classification": types,
                "Tampering Technique": reasons,
                "Intercepted Adversarial Text": payloads,
                "Assigned Action": c.get("verification_action", "Document security review"),
            })
        st.dataframe(sec_rows, hide_index=True, width="stretch")
        st.caption("All prompt injection attempts are automatically neutralized and stripped prior to NLP evaluation, preventing qualification score manipulation.")
    else:
        st.success("All applications in the current pool passed document integrity checks without security flags.")


# =============================================================================
# PAGE 2: HIRING FUNNEL & DECISION AUDIT
# =============================================================================
elif page == "Hiring Funnel & Decision Audit":
    heading("Hiring Funnel & Decision Audit", "Stage-by-stage audit: Review which applicants passed, which were flagged, and the exact reasons")
    terms_glossary_expander()

    stage_counts = {
        "Stage 1: Intake & Security": sum(f["stage_num"] == 1 for f in funnel_records),
        "Stage 2: Must-Haves Screen": sum(f["stage_num"] == 2 for f in funnel_records),
        "Stage 3: Population Screen": sum(f["stage_num"] == 3 for f in funnel_records),
        "Stage 4: Provenance Screen": sum(f["stage_num"] == 4 for f in funnel_records),
        "Stage 5: Final Routing": sum(f["stage_num"] == 5 for f in funnel_records),
    }
    kpi_cols = st.columns(5)
    for col, (stg, cnt) in zip(kpi_cols, stage_counts.items()):
        col.metric(stg, cnt)

    st.subheader("Filter Applications by Stage & Outcome")
    f1, f2, f3 = st.columns([2, 2, 2])
    stage_filter = f1.selectbox(
        "Filter by Stage Reached",
        ["All Stages", "Stage 1: Intake & Duplication / Security", "Stage 2: Must-Have Requirements", "Stage 3: Population Coordination", "Stage 4: History & Provenance", "Stage 5: Final Decision"],
    )
    outcome_filter = f2.selectbox(
        "Filter by Decision Outcome",
        [
            "All Outcomes",
            QUADRANT_NAMES["A"],
            QUADRANT_NAMES["B"],
            QUADRANT_NAMES["C"],
            QUADRANT_NAMES["D"],
            "Collapsed Duplicate",
            "Security Alert",
        ],
    )
    search_query = f3.text_input("Search Applicant Name or ID", placeholder="e.g. Jordan, C0007, Priya")

    filtered = funnel_records
    if stage_filter != "All Stages":
        stg_num = int(stage_filter[6:7])
        filtered = [f for f in filtered if f["stage_num"] == stg_num]
    if outcome_filter != "All Outcomes":
        if "Quadrant A" in outcome_filter:
            filtered = [f for f in filtered if f.get("quadrant") == "A"]
        elif "Quadrant B" in outcome_filter:
            filtered = [f for f in filtered if f.get("quadrant") == "B"]
        elif "Quadrant C" in outcome_filter:
            filtered = [f for f in filtered if f.get("quadrant") == "C"]
        elif "Quadrant D" in outcome_filter:
            filtered = [f for f in filtered if f.get("quadrant") == "D"]
        elif "Duplicate" in outcome_filter:
            filtered = [f for f in filtered if f.get("duplicate_of")]
        elif "Security" in outcome_filter:
            filtered = [f for f in filtered if f.get("injection_detected")]
    if search_query:
        filtered = [f for f in filtered if search_query.casefold() in (f["name"] + " " + f["candidate_id"]).casefold()]

    st.markdown(f"**Showing {len(filtered)} matching applications:**")
    display_funnel = []
    for f in filtered:
        q_key = f.get("quadrant", "C")
        display_funnel.append({
            "Candidate ID": f["candidate_id"],
            "Name": f["name"],
            "Stage Reached": f["stage_name"],
            "Screening Status": f["status"],
            "Reason / Trigger": f["reason"],
            "Scores (Q / P / A / C)": f"{f['Q']} / {f['P']} / {f['A']} / {f['C']}",
            "Decision Routing": QUADRANT_NAMES.get(q_key, f"Quadrant {q_key}"),
            "Action Required": f["action"],
        })
    st.dataframe(display_funnel, hide_index=True, width="stretch", height=420)

    if filtered:
        sel_cid = st.selectbox(
            "Select Candidate to Inspect Full Dossier",
            [f["candidate_id"] for f in filtered],
            format_func=lambda cid: next((f"{cid} · {f['name']} ({f['status']})" for f in filtered if f["candidate_id"] == cid), cid)
        )
        sel_cand = by_id.get(sel_cid)
        if sel_cand:
            candidate_profile(sel_cand, prefix="funnel")


# =============================================================================
# PAGE 3: CANDIDATE DOSSIER & PROFILE
# =============================================================================
elif page == "Candidate Dossier & Profile":
    heading("Candidate Dossier & Profile", "In-depth candidate review: 4-axis intelligence, claims verification, and work history")
    terms_glossary_expander()

    search_c1, search_c2 = st.columns([3, 1])
    query = search_c1.text_input("Search candidate by Name or ID", value="C0012")
    quad_select = search_c2.selectbox(
        "Filter by Quadrant",
        ["All Quadrants", QUADRANT_NAMES["A"], QUADRANT_NAMES["B"], QUADRANT_NAMES["C"], QUADRANT_NAMES["D"]]
    )

    matches = candidates
    if quad_select != "All Quadrants":
        q_letter = "A" if "Quadrant A" in quad_select else ("B" if "Quadrant B" in quad_select else ("C" if "Quadrant C" in quad_select else "D"))
        matches = [c for c in matches if c.get("quadrant") == q_letter]
    if query:
        matches = [c for c in matches if query.casefold() in (c["candidate_id"] + " " + c["name"] + " " + str(c.get("experience", ""))).casefold()]

    if not matches:
        st.warning("No candidate matched your search query.")
        matches = [by_id.get("C0012", candidates[0])]

    selected = st.selectbox(
        "Select Candidate Record",
        matches,
        format_func=lambda c: f"{c['candidate_id']} · {c['name']} (Q={c['Q']}, P={c['P']}, {QUADRANT_NAMES.get(c.get('quadrant', 'C'), c.get('quadrant'))}) — {c.get('verification_action', 'Review')}"
    )
    if selected:
        candidate_profile(selected, prefix="individual")


# =============================================================================
# PAGE 4: APPLICANT LANDSCAPE
# =============================================================================
elif page == "Applicant Landscape (2D Embedding)":
    heading("Applicant Landscape", "Population-level semantic clustering detects template reuse, cohort collisions, and farms")
    terms_glossary_expander()

    st.plotly_chart(landscape_figure(), width="stretch")

    flagged = [c for c in clusters if c.get("coordination_score", 0) >= 55]
    selected = st.selectbox(
        "Inspect Coordinated Population Group",
        flagged,
        format_func=lambda c: f"{c['display_name']} · Coordination Score: {c['coordination_score']}% · {c['size']} members"
    )
    if selected:
        st.info(selected["summary"])
        st.caption(f"Cluster #{selected['cluster']} contains {selected['size']} members with high wording collision. Review in the Cluster & Template Inspector.")


# =============================================================================
# PAGE 5: CLUSTER & TEMPLATE INSPECTOR
# =============================================================================
elif page == "Cluster & Template Inspector":
    heading("Cluster & Template Inspector", "Analyze shared phrasing, synchronized submission spikes, and portfolio domain collision")
    terms_glossary_expander()

    selected = st.selectbox("Population group", clusters, format_func=lambda c: f"{c['display_name']} · {c['size']} members (Coordination: {c['coordination_score']}%)")
    if selected:
        st.info(selected["summary"])

        features = pd.DataFrame({
            "Signal": ["Wording Similarity", "Shared Phrasing Ratio", "60-Min Submission Spike", "Shared Portfolio Domain", "Overall Coordination Score"],
            "Score (%)": [selected["mean_similarity"]*100, selected["shared_phrase_ratio"]*100, selected["submission_concentration"]*100, selected["shared_portfolio_domain_ratio"]*100, selected["coordination_score"]],
        })
        st.plotly_chart(px.bar(features, x="Score (%)", y="Signal", orientation="h", range_x=[0, 100], color="Score (%)", color_continuous_scale="Teal"), width="stretch")

        members = [by_id[cid] for cid in selected["member_ids"] if cid in by_id]
        st.subheader(f"Group Members ({len(members)} Applicants)")
        st.dataframe([{"ID": c["candidate_id"], "Name": c["name"], "Q": c["Q"], "P": c["P"], "A": c["A"], "C": c["C"], "Action": c["verification_action"]} for c in members], hide_index=True, width="stretch")

        st.subheader("Shared Wording & Template Collision Sample")
        cols = st.columns(min(2, len(members)))
        for col, candidate in zip(cols, members[:2]):
            col.markdown(f"**{candidate['name']} ({candidate['candidate_id']})**")
            col.markdown(highlighted(candidate["text"], selected["shared_phrases"]), unsafe_allow_html=True)


# =============================================================================
# PAGE 6: QUADRANT DECISION MATRIX
# =============================================================================
elif page == "Quadrant Decision Matrix":
    heading("Qualification Fit vs Trust Decision Matrix", "Four distinct operational quadrants ensuring no qualified candidate is auto-rejected")
    terms_glossary_expander()

    st.plotly_chart(quadrant_figure(candidates), width="stretch")

    st.subheader("Quadrant Descriptions & Operational Rules")
    q_cols = st.columns(4)
    for col, q_key in zip(q_cols, ["A", "B", "C", "D"]):
        col.markdown(f"**{QUADRANT_NAMES[q_key]}**")
        col.caption(QUADRANT_DESCRIPTIONS[q_key])


# =============================================================================
# PAGE 7: PRIORITY QUEUE & STEP-UP ROUTING
# =============================================================================
elif page == "Priority Queue & Step-Up Routing":
    heading("Priority Queue & Step-Up Routing", "High-qualification, verified applications first; specific doubts get specific human checks")
    terms_glossary_expander()

    top_tab, step_tab = st.tabs(["Top 100 Priority Queue (Quadrant A)", "Step-Up Verification Actions (Quadrant B)"])
    with top_tab:
        st.subheader("Quadrant A · Priority Fast-Track Candidates")
        top = [by_id[cid] for cid in scored.get("top_100", []) if cid in by_id]
        st.dataframe(
            [{"Rank": c.get("priority_rank", i+1), "ID": c["candidate_id"], "Name": c["name"], "Qualification (Q)": c["Q"], "Verified Proof (P)": c["P"], "Anomaly (A)": c["A"], "Trust Level": c.get("trust_confidence", "HIGH")} for i, c in enumerate(top)],
            hide_index=True, width="stretch", height=450
        )
    with step_tab:
        st.subheader("Quadrant B · Candidates Requiring Targeted Verification")
        step_cands = [c for c in candidates if c.get("quadrant") == "B"]
        st.dataframe(
            [{"Required Action": c.get("verification_action", "Verify"), "ID": c["candidate_id"], "Name": c["name"], "Qualification (Q)": c["Q"], "Trust Score": c.get("trust_score", 0), "Reason": determine_funnel_stage(c)["reason"]} for c in sorted(step_cands, key=lambda x: (x.get("verification_action", ""), -x["Q"]))],
            hide_index=True, width="stretch", height=450
        )


# =============================================================================
# PAGE 8: EXTERNAL DATASETS BENCHMARK
# =============================================================================
elif page == "External Datasets Benchmark":
    heading("External Resume Datasets Benchmark", "Test hybrid parsing and 4-axis intelligence across real-world datasets")
    terms_glossary_expander()

    bench_file = PROCESSED / "dataset_benchmark.json"
    top_c1, top_c2, top_c3 = st.columns([2, 1, 1])
    dataset_choice = top_c1.selectbox(
        "Select Dataset Source",
        ["all", "huggingface", "resumeatlas", "kaggle", "innovatiana"],
        format_func=lambda s: {
            "all": "All 4 Datasets Combined",
            "huggingface": "HuggingFace (datasetmaster/resumes)",
            "resumeatlas": "GitHub / ResumeAtlas (noran-mohamed)",
            "kaggle": "Kaggle PDFs (hadikp/resume-data-pdf)",
            "innovatiana": "Innovatiana Resume Dataset",
        }.get(s, s)
    )
    sample_limit = top_c2.slider("Record Limit", min_value=5, max_value=50, value=20, step=5)
    run_btn = top_c3.button("Run Benchmark Evaluation", type="primary")

    if run_btn or not bench_file.exists():
        with st.spinner(f"Evaluating dataset '{dataset_choice}'..."):
            from test_datasets import evaluate_dataset
            benchmark_data = evaluate_dataset(source=dataset_choice, limit=sample_limit)
            bench_file.write_text(json.dumps(benchmark_data, indent=2))
            st.success(f"Benchmark completed for {benchmark_data.get('total_evaluated', 0)} candidates in {benchmark_data.get('elapsed_seconds', 0)}s.")
    else:
        benchmark_data = json.loads(bench_file.read_text())

    if benchmark_data:
        m = benchmark_data.get("parser_metrics", {})
        mcols = st.columns(6)
        mcols[0].metric("Total Evaluated", benchmark_data.get("total_evaluated", 0))
        mcols[1].metric("Local Parse Rate", f"{m.get('local_tier_pct', 100)}%")
        mcols[2].metric("LLM Fallback Rate", f"{m.get('llm_tier_pct', 0)}%")
        mcols[3].metric("Name Yield", f"{m.get('name_yield_pct', 0)}%")
        mcols[4].metric("Experience Yield", f"{m.get('experience_yield_pct', 0)}%")
        mcols[5].metric("Avg Skills/Resume", m.get("avg_skills_per_resume", 0))

        left_b, right_b = st.columns([3, 2])
        q_dist = benchmark_data.get("quadrant_distribution", {})
        q_df = pd.DataFrame([{"Quadrant": QUADRANT_NAMES.get(k, f"Quadrant {k}"), "Count": v} for k, v in q_dist.items()])
        color_map = {
            QUADRANT_NAMES["A"]: "#0f766e",
            QUADRANT_NAMES["B"]: "#b45309",
            QUADRANT_NAMES["C"]: "#b91c1c",
            QUADRANT_NAMES["D"]: "#1d4ed8",
        }
        left_b.plotly_chart(
            px.bar(q_df, x="Quadrant", y="Count", color="Quadrant", color_discrete_map=color_map, title="Candidate Trust vs Qualification Distribution"),
            width="stretch"
        )

        act_dist = benchmark_data.get("top_verification_actions", {})
        act_df = pd.DataFrame([{"Action": k, "Count": v} for k, v in act_dist.items()])
        right_b.plotly_chart(px.pie(act_df, names="Action", values="Count", title="Recommended Verification Steps"), width="stretch")

        st.subheader("Evaluated Candidates Explorer")
        c_list = benchmark_data.get("candidates", [])
        if c_list:
            display_rows = [{
                "ID": c["candidate_id"],
                "Name": c["name"],
                "Dataset": c.get("dataset_source", ""),
                "Category": c.get("dataset_category", ""),
                "Q": c["Q"], "P": c["P"], "A": c["A"],
                "Trust Confidence": c["trust_confidence"],
                "Decision Routing": QUADRANT_NAMES.get(c["quadrant"], f"Quadrant {c['quadrant']}"),
                "Parser Tier": c.get("parser_tier", "local").upper(),
                "Confidence": f"{c.get('parser_confidence', 0)}%",
                "Next Step": c["verification_action"]
            } for c in c_list]
            st.dataframe(display_rows, hide_index=True, width="stretch")


# =============================================================================
# PAGE 9: LIVE RESUME PDF ANALYZER
# =============================================================================
else:
    heading("Live Resume PDF Analyzer", "Instant parsing, integrity sanitization, and 4-axis decision scoring for uploaded resumes")
    terms_glossary_expander()

    col1, col2 = st.columns([2, 1])
    with col1:
        upload = st.file_uploader("Upload a candidate PDF resume", type=["pdf"])
    with col2:
        st.markdown("**Extraction Settings**")
        with st.expander("GitHub & Parser Overrides", expanded=False):
            github_override = st.text_input(
                "Manual GitHub Username Override (Optional)",
                placeholder="e.g. torvalds",
                help="Leave blank to use the GitHub handle/URL auto-extracted from the PDF. Enter a username here only if testing demo resumes with synthetic/non-existent profiles."
            )
            force_llm = st.checkbox("Force deep LLM extraction (OpenAI GPT-4o-mini)", value=False)

    if upload:
        payload = upload.getvalue()
        digest = hashlib.sha256(payload).hexdigest()[:12]
        upload_dir = ROOT / "data/uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{digest}.pdf"
        path.write_bytes(payload)

        with st.status("Analyzing uploaded resume...", expanded=True) as status:
            st.write("Extracting text and parsing layout-aware sections...")
            st.write("Sanitizing instructions and checking hidden text...")
            from live import analyze_pdf
            result = analyze_pdf(path, github_override, force_llm=force_llm)
            st.write("Computing 2D population projection and cluster match...")
            st.write("Cross-referencing GitHub timeline and provenance claims...")
            st.write("Calculating 4-Axis scores and decision quadrant...")
            status.update(label="Analysis complete.", state="complete")

        gh_src = result.get("github_source", "none")
        gh_user = result.get("github_username", "")
        if gh_src == "auto_extracted":
            st.info(f"GitHub Profile: **@{gh_user}** (Auto-extracted from PDF link/text)")
        elif gh_src == "manual_override":
            st.info(f"GitHub Profile: **@{gh_user}** (Applied via manual test override)")
        else:
            st.info("No GitHub profile detected in resume. (Absence is treated as unverified, never penalized as negative evidence).")

        candidate_profile(result, prefix="live")

        st.subheader("Population & Decision Matrix Context")
        left, right = st.columns(2)
        left.plotly_chart(landscape_figure(result), width="stretch")
        right.plotly_chart(quadrant_figure(candidates, result), width="stretch")

st.divider()
st.caption("Signals prioritize human review. No candidate is auto-rejected. Designed as decision support with NYC AEDT-aware transparency and human oversight.")
