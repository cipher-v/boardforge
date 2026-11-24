from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from boardforge.connectors.registry import CONNECTORS
from boardforge.enrich import enrich_job
from boardforge.models import Job, SourceRef


DEV_CORPUS_COMPANIES = {"coinbase", "jumio", "twilio"}
CATEGORY_ORDER = (
    "ai_ml", "computer_vision", "backend", "data_engineering", "qa",
    "support", "management", "nontechnical",
)


class HoldoutSource(BaseModel):
    company: str
    kind: str
    key: str
    board_url: str


class HoldoutRecord(BaseModel):
    stable_id: str
    source_url: str
    source_board_url: str
    collected_at: str
    categories: list[str] = Field(default_factory=list)
    seniority: str
    job: dict[str, Any]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_sources(path: str | Path) -> list[HoldoutSource]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = [HoldoutSource.model_validate(item) for item in raw["sources"]]
    companies = [source.company.casefold() for source in sources]
    if len(sources) < 8 or len(sources) > 12:
        raise ValueError("holdout manifest must contain 8-12 sources")
    if len(companies) != len(set(companies)):
        raise ValueError("holdout manifest contains duplicate companies")
    overlap = sorted(set(companies) & DEV_CORPUS_COMPANIES)
    if overlap:
        raise ValueError(f"holdout companies overlap the development corpus: {overlap}")
    unsupported = sorted({source.kind for source in sources} - set(CONNECTORS))
    if unsupported:
        raise ValueError(f"unsupported holdout connectors: {unsupported}")
    return sources


_CATEGORY_PATTERNS = {
    "computer_vision": (
        r"computer vision|perception|image processing|visual (?:ai|recognition)|"
        r"object detection|autonomous vehicle|robotics",
    ),
    "ai_ml": (
        r"machine learning|\bai\b|artificial intelligence|deep learning|\bllm\b|"
        r"language model|generative ai|research scientist",
    ),
    "data_engineering": (
        r"data engineer|data platform|data pipeline|\betl\b|analytics engineer|"
        r"data infrastructure|warehouse",
    ),
    "qa": (
        r"quality assurance|\bqa\b|test (?:engineer|automation)|software.*test|"
        r"quality engineer|sdet",
    ),
    "support": (
        r"support engineer|customer support|technical support|customer success|"
        r"solutions engineer|technical account|customer experience",
    ),
    "management": (
        r"engineering manager|manager,? (?:engineering|software|data|machine learning)|"
        r"director|head of|vp |vice president|people manager",
    ),
    "backend": (
        r"backend|back-end|distributed systems|platform engineer|infrastructure engineer|"
        r"software engineer|api|server",
    ),
    "nontechnical": (
        r"recruit|talent|sales|account executive|marketing|legal|counsel|finance|"
        r"people operations|human resources|communications|product designer|workplace",
    ),
}

_TITLE_CATEGORY_ORDER = (
    "nontechnical", "support", "management", "qa", "data_engineering",
    "computer_vision", "ai_ml", "backend",
)


def classify_job(job: Job) -> list[str]:
    title = " ".join(job.title.casefold().split())
    title_categories = [
        category
        for category in _TITLE_CATEGORY_ORDER
        if any(re.search(pattern, title, re.I) for pattern in _CATEGORY_PATTERNS[category])
    ]
    if title_categories:
        return title_categories
    # Description fallback is deliberately limited to role-specific technical
    # language. Corporate boilerplate must not turn every role into AI/support.
    description = job.description.casefold()[:5000]
    description_categories = [
        category
        for category in (
            "computer_vision", "data_engineering", "qa", "ai_ml", "backend"
        )
        if any(
            re.search(pattern, description, re.I)
            for pattern in _CATEGORY_PATTERNS[category]
        )
    ]
    return description_categories or ["nontechnical"]


def classify_seniority(job: Job) -> str:
    title = job.title.casefold()
    text = f"{title} {job.employment_type.casefold()}"
    if re.search(r"intern|internship|co-op|apprentice", text):
        return "internship"
    if re.search(r"new grad|graduate|entry.level|junior|associate|university", text):
        return "entry_level"
    if re.search(r"senior|staff|principal|lead|manager|director|head|\bsr\.?\b", title):
        return "experienced"
    if job.min_experience_years is not None and job.min_experience_years >= 3:
        return "experienced"
    return "unspecified"


def _selection_key(job: Job) -> tuple[str, str, str]:
    return (job.title.casefold(), job.location.casefold(), job.stable_id)


def select_balanced_jobs(
    by_company: dict[str, list[Job]], target_count: int
) -> list[Job]:
    if not 80 <= target_count <= 120:
        raise ValueError("holdout target must be between 80 and 120 jobs")
    companies = sorted(by_company)
    if not companies:
        return []
    base, extra = divmod(target_count, len(companies))
    selected: list[Job] = []
    for company_index, company in enumerate(companies):
        quota = base + (1 if company_index < extra else 0)
        jobs = sorted(by_company[company], key=_selection_key)
        buckets: dict[str, list[Job]] = defaultdict(list)
        for job in jobs:
            buckets[classify_job(job)[0]].append(job)
        used: set[str] = set()
        company_selected: list[Job] = []
        rotated = CATEGORY_ORDER[company_index % len(CATEGORY_ORDER):] + CATEGORY_ORDER[:company_index % len(CATEGORY_ORDER)]
        while len(company_selected) < quota:
            added = False
            for category in rotated:
                candidate = next(
                    (job for job in buckets[category] if job.stable_id not in used),
                    None,
                )
                if candidate is not None:
                    company_selected.append(candidate)
                    used.add(candidate.stable_id)
                    added = True
                    if len(company_selected) == quota:
                        break
            if not added:
                break
        if len(company_selected) < quota:
            for job in jobs:
                if job.stable_id not in used:
                    company_selected.append(job)
                    used.add(job.stable_id)
                    if len(company_selected) == quota:
                        break
        selected.extend(company_selected)
    return sorted(selected, key=lambda job: (job.company.casefold(), *_selection_key(job)))


async def collect_holdout(
    sources_path: str | Path,
    output_path: str | Path,
    *,
    target_count: int = 100,
    force: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite frozen holdout corpus: {output}; use force explicitly"
        )
    sources = load_sources(sources_path)
    collected_at = datetime.now(timezone.utc).isoformat()

    async def fetch(source: HoldoutSource) -> tuple[HoldoutSource, list[Job]]:
        ref = SourceRef(kind=source.kind, key=source.key, board_url=source.board_url)
        jobs = await CONNECTORS[source.kind].fetch(ref)
        enriched = []
        for raw_job in jobs:
            job = enrich_job(raw_job).model_copy(update={"company": source.company})
            if job.description.strip() and job.apply_url.strip():
                enriched.append(job)
        return source, enriched

    fetched = await asyncio.gather(*(fetch(source) for source in sources))
    by_company = {source.company: jobs for source, jobs in fetched}
    too_small = {company: len(jobs) for company, jobs in by_company.items() if len(jobs) < 5}
    if too_small:
        raise ValueError(f"holdout sources returned too few usable jobs: {too_small}")
    selected = select_balanced_jobs(by_company, target_count)
    if len(selected) != target_count:
        raise ValueError(f"could only select {len(selected)} of {target_count} jobs")

    source_by_key = {(source.kind, source.key): source for source in sources}
    records = []
    for job in selected:
        source = source_by_key[(job.source_kind, job.source_key)]
        records.append(HoldoutRecord(
            stable_id=job.stable_id,
            source_url=job.apply_url,
            source_board_url=source.board_url,
            collected_at=collected_at,
            categories=classify_job(job),
            seniority=classify_seniority(job),
            job=job.model_dump(mode="json"),
        ).model_dump(mode="json"))

    payload = {
        "version": "1.0",
        "kind": "boardforge_independent_holdout_corpus",
        "status": "frozen",
        "captured_at": collected_at,
        "selection": {
            "algorithm": "deterministic_company_quota_category_round_robin_v1",
            "target_jobs": target_count,
            "development_companies_excluded": sorted(DEV_CORPUS_COMPANIES),
        },
        "sources": [source.model_dump() for source in sources],
        "records": records,
    }
    payload["records_sha256"] = canonical_sha256(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output": str(output),
        "jobs": len(records),
        "companies": len(sources),
        "categories": dict(Counter(c for record in records for c in record["categories"])),
        "seniority": dict(Counter(record["seniority"] for record in records)),
        "records_sha256": payload["records_sha256"],
        "paid_api_calls": 0,
    }
