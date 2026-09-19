from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).parent
RAW = ROOT / "data/raw/resumes.json"
OUT = ROOT / "data/processed/integrity.json"
INJECTION_RE = re.compile(
    r"(?i)(ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|directions)[^.\n]*[.]?|"
    r"system\s*:[^.\n]*[.]?|rank\s+this\s+candidate[^.\n]*[.]?|"
    r"you\s+are\s+an\s+ai[^.\n]*[.]?|select\s+this\s+applicant[^.\n]*[.]?)"
)


def color_rgb(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def inspect_pdf(path: Path) -> tuple[list[dict], str]:
    hidden, text_parts = [], []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc):
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        text_parts.append(text)
                        reasons = []
                        if all(v > 240 for v in color_rgb(int(span.get("color", 0)))):
                            reasons.append("white text on white background")
                        if float(span.get("size", 12)) < 4:
                            reasons.append("font size below 4pt")
                        rect = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                        if not page.rect.contains(rect):
                            reasons.append("text positioned outside page bounds")
                        if reasons:
                            hidden.append({"text": text, "reason": "; ".join(reasons), "page": page_no + 1})
    return hidden, "\n".join(text_parts)


def sanitize(text: str) -> str:
    cleaned = INJECTION_RE.sub("[security instruction removed]", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def analyze_record(rec: dict) -> dict:
    hidden, pdf_text = [], ""
    if rec.get("pdf_path"):
        hidden, pdf_text = inspect_pdf(ROOT / rec["pdf_path"])
    matches = [m.group(0).strip() for m in INJECTION_RE.finditer("\n".join([rec.get("text", ""), pdf_text]))]
    events = []
    for item in hidden:
        if INJECTION_RE.search(item["text"]):
            events.append({"type": "prompt_injection", "severity": "high", "text": item["text"], "reason": item["reason"]})
        else:
            events.append({"type": "hidden_text", "severity": "medium", "text": item["text"], "reason": item["reason"]})
    for match in matches:
        if not any(e["text"] == match for e in events):
            events.append({"type": "prompt_injection", "severity": "high", "text": match, "reason": "instruction-like text matched a security pattern"})
    return {
        "candidate_id": rec["candidate_id"], "hidden_spans": hidden,
        "injection_detected": bool(matches), "sanitized_text": sanitize(rec.get("text", "")),
        "security_events": events, "duplicate_of": None,
    }


def main() -> None:
    records = json.loads(RAW.read_text())
    output, seen = [], {}
    for rec in records:
        result = analyze_record(rec)
        digest = normalized_hash(result["sanitized_text"])
        result["duplicate_of"] = seen.get(digest)
        seen.setdefault(digest, rec["candidate_id"])
        output.append(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2))
    print(f"Integrity: {sum(x['injection_detected'] for x in output)} injections, "
          f"{sum(bool(x['hidden_spans']) for x in output)} PDFs with hidden spans, "
          f"{sum(bool(x['duplicate_of']) for x in output)} duplicate applications")


if __name__ == "__main__":
    main()
