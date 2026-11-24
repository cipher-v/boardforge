from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boardforge.holdout import canonical_sha256
from boardforge.matching import hard_violations
from boardforge.models import Job, Preferences


def _field(job: Job, name: str) -> str:
    if name == "title":
        return job.title
    if name == "location":
        return job.location
    if name == "description":
        return job.description
    if name == "text":
        return f"{job.title}\n{job.description}"
    raise ValueError(f"unsupported label criterion field: {name}")


def _criterion(job: Job, rule: dict[str, Any]) -> tuple[bool, dict[str, str] | None]:
    value = _field(job, rule.get("field", "text"))
    for pattern in rule["any"]:
        match = re.search(pattern, value, re.I)
        if match:
            start = max(0, match.start() - 90)
            end = min(len(value), match.end() + 140)
            snippet = " ".join(value[start:end].split())
            return True, {
                "field": rule.get("field", "text"),
                "pattern": pattern,
                "matched": match.group(0),
                "snippet": snippet,
            }
    return False, None


def _all_criteria(
    job: Job, rules: list[dict[str, Any]]
) -> tuple[bool, list[dict[str, str]]]:
    evidence: list[dict[str, str]] = []
    for rule in rules:
        matched, item = _criterion(job, rule)
        if not matched:
            return False, []
        if item:
            evidence.append(item)
    return bool(rules), evidence


def _any_criterion(
    job: Job, rules: list[dict[str, Any]]
) -> tuple[bool, list[dict[str, str]]]:
    for rule in rules:
        matched, item = _criterion(job, rule)
        if matched:
            return True, [item] if item else []
    return False, []


def label_holdout(
    corpus_path: str | Path,
    intents_path: str | Path,
    output_path: str | Path,
    review_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    review = Path(review_path)
    if (output.exists() or review.exists()) and not force:
        raise FileExistsError("refusing to overwrite frozen holdout labels/review export")
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    intents = json.loads(Path(intents_path).read_text(encoding="utf-8"))
    if canonical_sha256(corpus["records"]) != corpus["records_sha256"]:
        raise ValueError("holdout corpus checksum mismatch")
    cases = intents["cases"]
    if not 30 <= len(cases) <= 40:
        raise ValueError("holdout must contain 30-40 search intents")
    names = [case["name"] for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("holdout intent names must be unique")

    jobs = [Job.model_validate(record["job"]) for record in corpus["records"]]
    labeled_cases = []
    totals: Counter[str] = Counter()
    for case in cases:
        # Labels deliberately use only explicit structured constraints from the
        # frozen intent spec. Production query parsing/ranking is not consulted.
        preferences = Preferences.model_validate(case.get("saved_preferences", {}))
        labels: dict[str, dict[str, Any]] = {}
        for job in jobs:
            violations = hard_violations(job, preferences)
            forbidden, forbidden_evidence = _any_criterion(
                job, case.get("forbidden", [])
            )
            direct, direct_evidence = _all_criteria(job, case.get("direct", []))
            adjacent, adjacent_evidence = _all_criteria(
                job, case.get("adjacent", [])
            )
            if violations or forbidden:
                label = "forbidden"
                evidence: list[dict[str, Any]] = forbidden_evidence
                evidence.extend(
                    {"field": "structured", "matched": violation, "snippet": "hard constraint violation"}
                    for violation in violations
                )
            elif direct:
                label, evidence = "direct", direct_evidence
            elif adjacent:
                label, evidence = "adjacent", adjacent_evidence
            else:
                label, evidence = "irrelevant", []
            totals[label] += 1
            labels[job.stable_id] = {"label": label, "evidence": evidence}
        labeled_cases.append({
            "name": case["name"],
            "family": case["family"],
            "query": case["query"],
            "saved_preferences": case.get("saved_preferences", {}),
            "intent_note": case["intent_note"],
            "criteria": {
                "direct": case.get("direct", []),
                "adjacent": case.get("adjacent", []),
                "forbidden": case.get("forbidden", []),
            },
            "labels": labels,
        })

    frozen_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "version": "1.0-provisional",
        "kind": "boardforge_independent_holdout_labels",
        "status": "frozen_provisional",
        "verification_status": "automated_provisional_not_human_verified",
        "warning": "Do not treat these labels as human-verified. Review the evidence export and record human decisions in a new version.",
        "method": "explicit intent-specific criteria evaluated against frozen descriptions before ranking",
        "frozen_at": frozen_at,
        "corpus_records_sha256": corpus["records_sha256"],
        "intents_sha256": canonical_sha256(cases),
        "cases": labeled_cases,
    }
    payload["labels_sha256"] = canonical_sha256(labeled_cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Holdout v1 provisional-label review",
        "",
        "> Status: **automated provisional — not human verified**. Review decisions must be saved as a new dataset version; do not edit the frozen v1 corpus.",
        "",
        f"- Corpus checksum: `{corpus['records_sha256']}`",
        f"- Label checksum: `{payload['labels_sha256']}`",
        f"- Frozen at: {frozen_at}",
        f"- Intents: {len(labeled_cases)}; jobs: {len(jobs)}; decisions: {len(labeled_cases) * len(jobs)}",
        "",
    ]
    for case in labeled_cases:
        positives = [
            (stable_id, value) for stable_id, value in case["labels"].items()
            if value["label"] != "irrelevant"
        ]
        lines.extend([
            f"## {case['name']} ({case['family']})",
            "",
            f"Query: `{case['query']}`",
            "",
            case["intent_note"],
            "",
        ])
        if not positives:
            lines.extend(["No provisional positive or forbidden labels (intentional abstention case).", ""])
            continue
        for stable_id, value in positives:
            snippets = "; ".join(
                evidence.get("snippet", evidence.get("matched", ""))
                for evidence in value["evidence"]
            )
            lines.append(f"- **{value['label']}** `{stable_id}` — {snippets[:500]}")
        lines.append("")
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text("\n".join(lines), encoding="utf-8")
    return {
        "labels": str(output), "review": str(review), "cases": len(cases),
        "jobs": len(jobs), "counts": dict(totals),
        "verification_status": payload["verification_status"],
        "labels_sha256": payload["labels_sha256"],
    }
