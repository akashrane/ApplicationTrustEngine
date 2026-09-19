from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data/processed"

st.set_page_config(page_title="Applicant Trust Intelligence", page_icon="◈", layout="wide")
st.markdown("""
<style>
:root { --ink:#102a43; --teal:#0f766e; --amber:#d97706; --paper:#f7f9fc; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stSidebar"] { background:#0b2138; }
[data-testid="stSidebar"] * { color:#f8fafc !important; }
.hero {padding:1.2rem 1.4rem;border-radius:18px;background:linear-gradient(120deg,#102a43,#0f766e);color:white;margin-bottom:1rem}
.hero h1 {margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;color:#dbeafe}
.badge {display:inline-block;padding:.25rem .65rem;border-radius:999px;font-weight:700;font-size:.78rem;letter-spacing:.04em}
.HIGH{background:#d1fae5;color:#065f46}.MEDIUM{background:#fef3c7;color:#92400e}.LOW{background:#fee2e2;color:#991b1b}.UNKNOWN{background:#e2e8f0;color:#334155}
.challenge {border:1px solid #cbd5e1;border-radius:16px;padding:1rem;background:white}
mark {background:#fde68a;padding:0 .1rem}.small-note{color:#64748b;font-size:.88rem}
[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;padding:1rem;border-radius:14px}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data() -> tuple[dict, dict]:
    return json.loads((PROCESSED / "scored.json").read_text()), json.loads((PROCESSED / "population.json").read_text())


try:
    scored, population = load_data()
except FileNotFoundError:
    st.error("Processed data is missing. Run `make all` first.")
    st.stop()

candidates = scored["candidates"]
by_id = {c["candidate_id"]: c for c in candidates}
active = [c for c in candidates if not c["duplicate_of"]]
clusters = population["clusters"]


def heading(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def gauge(value: int, label: str, color: str, inverse: bool = False) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": label}, gauge={
        "axis": {"range": [0, 100]}, "bar": {"color": color}, "bgcolor": "#e2e8f0",
        "steps": ([{"range": [0, 35], "color": "#d1fae5"}, {"range": [70, 100], "color": "#fee2e2"}] if inverse else
                  [{"range": [0, 40], "color": "#fee2e2"}, {"range": [75, 100], "color": "#d1fae5"}]),
    }))
    fig.update_layout(height=210, margin=dict(l=18, r=18, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def score_row(candidate: dict) -> None:
    cols = st.columns(4)
    for col, key, color, inverse in zip(cols, ["Q", "P", "A", "C"], ["#0f766e", "#2563eb", "#d97706", "#7c3aed"], [False, False, True, True]):
        col.plotly_chart(gauge(candidate[key], {"Q": "Qualification", "P": "Provenance", "A": "Anomaly", "C": "Coordination"}[key], color, inverse), use_container_width=True, key=f"g-{candidate['candidate_id']}-{key}")


def landscape_figure(extra: dict | None = None) -> go.Figure:
    rows = [{**c, "cluster_color": c["cluster_name"] if c["cluster"] != -1 else "Noise"} for c in active]
    frame = pd.DataFrame(rows)
    fig = px.scatter(frame, x="x", y="y", color="cluster_color", hover_name="name",
                     hover_data={"candidate_id": True, "Q": True, "P": True, "A": True, "C": True, "x": False, "y": False},
                     color_discrete_sequence=px.colors.qualitative.Safe, opacity=.72)
    fig.update_traces(marker={"size": 7})
    if extra:
        fig.add_trace(go.Scatter(x=[extra["x"]], y=[extra["y"]], mode="markers", name="Uploaded résumé",
                                 marker={"symbol": "star", "size": 22, "color": "#ef4444", "line": {"width": 2, "color": "white"}}, text=[extra["name"]]))
    fig.update_layout(height=610, legend_title="Population group", xaxis_title="UMAP 1", yaxis_title="UMAP 2", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def quadrant_figure(rows: list[dict], extra: dict | None = None) -> go.Figure:
    frame = pd.DataFrame([c for c in rows if not c["duplicate_of"]])
    fig = px.scatter(frame, x="trust_score", y="Q", color="quadrant", hover_name="name",
                     hover_data=["candidate_id", "P", "A", "C", "verification_action"],
                     color_discrete_map={"A": "#0f766e", "B": "#d97706", "C": "#dc2626", "D": "#2563eb"}, opacity=.7)
    if extra:
        fig.add_trace(go.Scatter(x=[extra["trust_score"]], y=[extra["Q"]], mode="markers", name="Uploaded résumé", marker={"symbol": "star", "size": 22, "color": "#111827"}))
    fig.add_vline(x=60, line_dash="dash", line_color="#64748b"); fig.add_hline(y=70, line_dash="dash", line_color="#64748b")
    for x, y, label in [(80, 92, "A · Priority review"), (30, 92, "B · Step-up verify"), (30, 20, "C · Lower fit / review"), (80, 20, "D · Authentic, wrong role")]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, bgcolor="rgba(255,255,255,.8)")
    fig.update_layout(height=610, xaxis_title="Trust confidence index", yaxis_title="Qualification", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def highlighted(text: str, phrases: list[str]) -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for phrase in sorted(phrases[:8], key=len, reverse=True):
        safe = re.sub(re.escape(phrase), lambda m: f"<mark>{m.group(0)}</mark>", safe, flags=re.I)
    return safe.replace("\n", "<br>")


def candidate_profile(candidate: dict, prefix: str = "profile") -> None:
    left, right = st.columns([3, 1])
    left.subheader(f"{candidate['name']} · {candidate['candidate_id']}")
    left.caption(f"{candidate['experience'][0]['title']} · submitted {candidate['submitted_at'][:10]}")
    right.markdown(f'<span class="badge {candidate["trust_confidence"]}">{candidate["trust_confidence"]} TRUST</span>', unsafe_allow_html=True)
    score_row(candidate)
    tabs = st.tabs(["Role fit", "Claim evidence", "GitHub timeline", "Security", "Trust Challenge"])
    with tabs[0]:
        table = [{"Result": "✓" if r["met"] else "○", "Requirement": r["requirement"], "Type": r["type"]} for r in candidate["requirements"]]
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.info(f"Recommended next step: **{candidate['verification_action']}**")
    with tabs[1]:
        st.dataframe(candidate["claims"], hide_index=True, use_container_width=True,
                     column_config={"status": st.column_config.TextColumn("Status"), "evidence": st.column_config.TextColumn("Evidence", width="large")})
    with tabs[2]:
        timeline = pd.DataFrame(candidate["github_timeline"])
        if timeline.empty:
            st.caption("No linked timeline is available; this is unverified, not negative evidence.")
        else:
            timeline["date"] = pd.to_datetime(timeline["date"])
            fig = px.scatter(timeline, x="date", y="kind", color="kind", hover_name="event", hover_data=["language"] if "language" in timeline else None)
            fig.update_traces(marker={"size": 13}); fig.update_layout(height=330, yaxis_title="Evidence type")
            st.plotly_chart(fig, use_container_width=True, key=f"timeline-{prefix}-{candidate['candidate_id']}")
    with tabs[3]:
        if not candidate["security_events"]:
            st.success("No document security signal detected.")
        for event in candidate["security_events"]:
            st.warning(f"**{event['type'].replace('_',' ').title()}** — {event['reason']}\n\nExact extracted text: `{event['text']}`")
        st.caption("Qualification is calculated from sanitized text, so document security signals do not reduce Q.")
    with tabs[4]:
        st.markdown('<div class="challenge"><b>We couldn\'t verify one part of your application automatically.</b><br><span class="small-note">Choose the easiest way to add context. Skipping keeps your application in human review.</span></div>', unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].button("Verify work email", key=f"email-{prefix}"); cols[1].button("Connect GitHub", key=f"github-{prefix}")
        cols[2].button("Provide documentation", key=f"docs-{prefix}"); cols[3].button("Skip, recruiter will review", key=f"skip-{prefix}")


pages = ["Overview", "Applicant Landscape", "Cluster Inspector", "Candidate Profile", "Quadrant View", "Priority Queue", "Analyze a Résumé"]
st.sidebar.markdown("## ◈ Applicant Trust")
page = st.sidebar.radio("Workspace", pages)
st.sidebar.caption(f"Offline model · {scored['embedding_method']}")

if page == "Overview":
    heading("Applicant Trust Intelligence", "Population-aware signals for faster, evidence-centered human review")
    metrics = st.columns(5)
    for col, label, value in zip(metrics, ["Applicants", "Top 100", "Needs verification", "Security events", "Flagged clusters"],
                                 [len(candidates), len(scored["top_100"]), scored["counts"]["step_up"], scored["counts"]["security_events"], sum(c["coordination_score"] >= 55 for c in clusters)]):
        col.metric(label, value)
    left, right = st.columns([3, 2])
    funnel = pd.DataFrame(scored["funnel"])
    left.plotly_chart(px.funnel(funnel, x="value", y="stage", color_discrete_sequence=["#0f766e"]), use_container_width=True)
    right.subheader("How to read the system")
    right.markdown("**Q** asks whether the résumé meets the role. **P** shows what the linked history supports. **A** ranks unusual signals. **C** describes population-level coordination. They stay separate so reviewers can see why a case needs attention.")
    right.info("Start the demo with Priya Nair for strong supported evidence, then Jordan Reyes for step-up verification.")

elif page == "Applicant Landscape":
    heading("Applicant Landscape", "Patterns become visible when applications are evaluated as a population")
    st.plotly_chart(landscape_figure(), use_container_width=True)
    flagged = [c for c in clusters if c["coordination_score"] >= 55]
    selected = st.selectbox("Inspect a population group", flagged, format_func=lambda c: f"{c['display_name']} · C {c['coordination_score']} · {c['size']} members")
    if selected:
        st.info(selected["summary"])

elif page == "Cluster Inspector":
    heading("Cluster Inspector", "Review the shared evidence and keep plausible benign explanations in view")
    selected = st.selectbox("Population group", clusters, format_func=lambda c: f"{c['display_name']} · {c['size']} members")
    st.info(selected["summary"])
    features = pd.DataFrame({"Signal": ["Wording similarity", "Shared phrases", "60-minute concentration", "Shared portfolio domain", "Coordination score"],
                             "Value": [selected["mean_similarity"]*100, selected["shared_phrase_ratio"]*100, selected["submission_concentration"]*100, selected["shared_portfolio_domain_ratio"]*100, selected["coordination_score"]]})
    st.plotly_chart(px.bar(features, x="Value", y="Signal", orientation="h", range_x=[0, 100], color="Value", color_continuous_scale="Teal"), use_container_width=True)
    members = [by_id[cid] for cid in selected["member_ids"]]
    st.dataframe([{"ID": c["candidate_id"], "Name": c["name"], "Q": c["Q"], "P": c["P"], "A": c["A"], "C": c["C"], "Next step": c["verification_action"]} for c in members], hide_index=True, use_container_width=True)
    st.subheader("Shared wording sample")
    cols = st.columns(2)
    for col, candidate in zip(cols, members[:2]):
        col.markdown(f"**{candidate['name']}**")
        col.markdown(highlighted(candidate["text"], selected["shared_phrases"]), unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3); b1.button("Inspect"); b2.button("Request verification"); b3.button("Dismiss signal")

elif page == "Candidate Profile":
    heading("Candidate Profile", "Four distinct scores, the evidence behind them, and the minimum necessary next step")
    query = st.text_input("Search by candidate name or ID", value="C0012")
    matches = [c for c in candidates if query.casefold() in (c["candidate_id"] + " " + c["name"]).casefold()] or [by_id["C0012"]]
    selected = st.selectbox("Candidate", matches, format_func=lambda c: f"{c['candidate_id']} · {c['name']}")
    candidate_profile(selected)

elif page == "Quadrant View":
    heading("Qualification × Trust", "Qualified candidates with unresolved evidence move to step-up verification—not an automated outcome")
    st.plotly_chart(quadrant_figure(candidates), use_container_width=True)

elif page == "Priority Queue":
    heading("Priority Queue", "High-qualification, high-confidence applications first; specific doubts get specific checks")
    st.subheader("Top 100 priority review")
    top = [by_id[cid] for cid in scored["top_100"]]
    st.dataframe([{"Rank": c["priority_rank"], "ID": c["candidate_id"], "Name": c["name"], "Q": c["Q"], "P": c["P"], "A": c["A"], "C": c["C"]} for c in top], hide_index=True, use_container_width=True, height=430)
    st.subheader("Step-up verification")
    step = [c for c in candidates if c["quadrant"] == "B"]
    st.dataframe([{"Action": c["verification_action"], "ID": c["candidate_id"], "Name": c["name"], "Q": c["Q"], "Trust": c["trust_confidence"]} for c in sorted(step, key=lambda c: (c["verification_action"], -c["Q"]))], hide_index=True, use_container_width=True)

else:
    heading("Analyze a Résumé", "Score one PDF against the offline population; optional parsing and GitHub checks run only here")
    upload = st.file_uploader("Upload a PDF résumé", type=["pdf"])
    github_override = st.text_input("Optional: live-check a GitHub username")
    if upload:
        payload = upload.getvalue(); digest = hashlib.sha256(payload).hexdigest()[:12]
        upload_dir = ROOT / "data/uploads"; upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{digest}.pdf"; path.write_bytes(payload)
        with st.status("Analyzing résumé", expanded=True) as status:
            st.write("✓ Parsing and field extraction"); st.write("✓ Integrity and instruction sanitization")
            from live import analyze_pdf
            result = analyze_pdf(path, github_override)
            st.write("✓ Population similarity and cluster assignment"); st.write("✓ Claim provenance"); st.write("✓ Decision scoring")
            status.update(label="Analysis complete", state="complete")
        candidate_profile(result, "live")
        st.subheader("Population context")
        left, right = st.columns(2)
        left.plotly_chart(landscape_figure(result), use_container_width=True)
        right.plotly_chart(quadrant_figure(candidates, result), use_container_width=True)
        st.dataframe(result["similar_candidates"], hide_index=True, use_container_width=True)

st.divider()
st.caption("Signals prioritize human review. No candidate is auto-rejected. Designed as decision support with NYC AEDT-aware transparency and human oversight.")
