from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from boardforge.holdout import canonical_sha256
from boardforge.models import Job


# These are explicit semantic judgments over the frozen descriptions. IDs are
# external ATS IDs resolved against the corpus, not benchmark-specific ranking
# rules. Absence means irrelevant, except for the explicit constraint overrides
# below. Regex is not used to decide relevance.
SEMANTIC_DECISIONS: dict[str, dict[str, Any]] = {
    "broad_ml_engineer": {
        "direct": ["8242623"],
        "adjacent": ["5468939008", "5707966004", "ff1029bd-bb6d-4d78-a03e-5f9744d0b798", "b39ceb08-a0a7-4f8b-a760-2fb88e209956", "8242621", "8384186", "8242619"],
        "rationale": "Hands-on ML engineering is direct; applied science, AI delivery, and data-science roles are adjacent when ML is a core responsibility.",
    },
    "llm_inference_engineer": {
        "direct": [], "adjacent": ["8242623"],
        "rationale": "No posting owns LLM inference or serving optimization; the production MLE is adjacent, while generic LLM experimentation is incidental.",
    },
    "ai_research_scientist": {
        "direct": ["5468939008"], "adjacent": ["5707966004", "8242621", "8242623"],
        "rationale": "Anthropic explicitly trains Claude for scientific reasoning; broader applied AI and research-program roles are adjacent to foundation-model research.",
    },
    "ml_platform": {
        "direct": [], "adjacent": ["8242623"],
        "rationale": "No role owns a training/serving platform. Roblox MLE builds scalable ML infrastructure but the posting is product-model focused.",
    },
    "computer_vision_perception": {
        "direct": [], "adjacent": ["6219995004", "74aed644-d963-4c5d-b7a0-149bc35f9c50", "8242621", "8242623", "8386080", "8204785"],
        "rationale": "Graphics, mixed reality, 3D research, and multimodal ML neighbor vision work, but none assigns substantive perception/CV ownership.",
    },
    "multimodal_models": {
        "direct": ["8242623"], "adjacent": ["8242621"],
        "rationale": "The MLE explicitly develops multimodal architectures; the applied-science program includes multimodality but assignment is not guaranteed.",
    },
    "backend_general": {
        "direct": ["8401112", "8105716", "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184", "7bfa613e-151c-469b-9973-c89ee3d14838"],
        "adjacent": ["5073998008", "6233040004", "30a5c739-185f-4a55-8a63-85974d687444", "3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c", "a753a9e7-a361-426b-9c25-3cf2488c1730", "34413f8d-26bf-4bbc-8ade-eb309a0e2245", "8072244"],
        "rationale": "Backend services and infrastructure ownership are direct; CI, integration, business systems, security infrastructure, and broad software roles are adjacent.",
    },
    "distributed_systems": {
        "direct": ["8401112", "8105716", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184"],
        "adjacent": ["8378768", "8350419", "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "7bfa613e-151c-469b-9973-c89ee3d14838", "8242623", "8072244"],
        "rationale": "Direct roles own distributed/control-plane/crawler systems. Managers, general backend, ML infrastructure, and broad scale engineering are adjacent.",
    },
    "api_backend": {
        "direct": ["8401112"],
        "adjacent": ["8105716", "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184", "7bfa613e-151c-469b-9973-c89ee3d14838"],
        "rationale": "Cloudflare explicitly owns customer configuration APIs; other roles are substantive backend but do not make API implementation the stated task.",
    },
    "search_backend": {
        "direct": ["be94e89b-89d5-4f2a-a58b-7929c8d97f92", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184"], "adjacent": [],
        "rationale": "Both Perplexity roles explicitly build search infrastructure or crawler/indexing systems; market research is a different occupation.",
    },
    "data_engineer": {
        "direct": ["6233040004", "a753a9e7-a361-426b-9c25-3cf2488c1730"],
        "adjacent": ["e9bf7ffa-9cf2-4360-a2e4-e9fa66859927", "30a5c739-185f-4a55-8a63-85974d687444", "e916f0c8-5a31-4bb6-b4e3-7c4a64b95fda", "e577622f-6657-4e53-8941-b3a774b04448"],
        "rationale": "Figma and Palantir own production data pipelines. Data-platform leadership, business integrations, GTM data movement, and analytics are adjacent.",
    },
    "streaming_pipelines": {
        "direct": [], "adjacent": ["8378768", "e9bf7ffa-9cf2-4360-a2e4-e9fa66859927", "6233040004", "a753a9e7-a361-426b-9c25-3cf2488c1730"],
        "rationale": "Pipeline roles are adjacent, but no IC posting assigns Kafka/stream-processing ownership as the stated responsibility.",
    },
    "analytics_engineer": {
        "direct": ["e577622f-6657-4e53-8941-b3a774b04448"],
        "adjacent": ["6238480004", "6233040004", "30a5c739-185f-4a55-8a63-85974d687444", "e916f0c8-5a31-4bb6-b4e3-7c4a64b95fda", "8384186", "8242619"],
        "rationale": "Ramp's role is substantively full-stack analytics; research, integration, warehouse operations, and data science roles are adjacent.",
    },
    "data_plumbing_paraphrase": {
        "direct": ["6233040004", "30a5c739-185f-4a55-8a63-85974d687444", "a753a9e7-a361-426b-9c25-3cf2488c1730", "e916f0c8-5a31-4bb6-b4e3-7c4a64b95fda"],
        "adjacent": ["8378768", "e9bf7ffa-9cf2-4360-a2e4-e9fa66859927", "e577622f-6657-4e53-8941-b3a774b04448"],
        "rationale": "The direct roles explicitly move, integrate, route, or transform company data; platform leadership and analytics are adjacent.",
    },
    "qa_automation": {
        "direct": ["4b5e02a8-6a94-494d-9d49-037f2cfc144d"], "adjacent": ["5073998008"],
        "rationale": "Perplexity owns product QA; Anthropic owns automated CI/test infrastructure. Incidental testing in other engineering roles is irrelevant.",
    },
    "sdet_frameworks": {
        "direct": [], "adjacent": ["5073998008"],
        "rationale": "No posting is an SDET or owns product test frameworks; CI infrastructure is the nearest substantive neighboring responsibility.",
    },
    "technical_support": {
        "direct": ["00c2c97b-8514-4617-9883-e53e486b6dcd", "4cba9c95-d16f-440d-83e7-2352480f689f"],
        "adjacent": ["d8890a6d-ce76-4c3d-aacc-34c1ed150d91", "6181922004", "ff294bf0-7dc1-48d6-876d-25acc8de264e", "4967b1cf-a1f7-417b-b650-feebbff767a3", "8286666"],
        "rationale": "Palantir explicitly debugs customer technical issues. Solutions, adoption architecture, field engineering, and support management are adjacent.",
    },
    "customer_success": {
        "direct": ["5424443008", "8301102", "6104028004", "6232584004", "0dbd1790-2652-49fc-861f-7451ae4eb1cf", "ff294bf0-7dc1-48d6-876d-25acc8de264e", "4967b1cf-a1f7-417b-b650-feebbff767a3", "00c2c97b-8514-4617-9883-e53e486b6dcd", "4cba9c95-d16f-440d-83e7-2352480f689f", "6b2da750-51fb-4428-b682-fe2296d9aef7"],
        "adjacent": ["5426627008", "d8890a6d-ce76-4c3d-aacc-34c1ed150d91", "6181922004", "8286666"],
        "rationale": "Customer-success and enablement ownership is direct; GTM enablement, solutions/field engineering, and incident-support management are adjacent.",
    },
    "solutions_engineer": {
        "direct": ["d8890a6d-ce76-4c3d-aacc-34c1ed150d91", "6181922004"],
        "adjacent": ["8130580", "ff1029bd-bb6d-4d78-a03e-5f9744d0b798", "00c2c97b-8514-4617-9883-e53e486b6dcd", "4cba9c95-d16f-440d-83e7-2352480f689f", "8efd3a0a-fc66-46e2-9415-bffba10e2919"],
        "rationale": "Linear and Figma own enterprise technical solution work; sales engineering, AI delivery, enablement, and quota-carrying technical sales are adjacent.",
    },
    "engineering_manager": {
        "direct": ["8378768", "8350419", "e9bf7ffa-9cf2-4360-a2e4-e9fa66859927"],
        "adjacent": ["8261861", "8402154"],
        "rationale": "Direct roles lead backend/data/networking teams. Developer-tooling/SDLC engineering management is adjacent; IC and mobile-product roles are different intent.",
    },
    "backend_ic_not_manager": {
        "direct": ["8401112", "8105716", "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184", "7bfa613e-151c-469b-9973-c89ee3d14838"],
        "adjacent": ["5073998008", "6233040004", "30a5c739-185f-4a55-8a63-85974d687444", "3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c", "a753a9e7-a361-426b-9c25-3cf2488c1730", "34413f8d-26bf-4bbc-8ade-eb309a0e2245", "8072244"],
        "forbidden": ["8402154", "8378768", "8350419", "8261861", "e9bf7ffa-9cf2-4360-a2e4-e9fa66859927"],
        "rationale": "Backend IC responsibilities follow the broad backend case; only otherwise related people-management roles are explicitly forbidden by 'no management'.",
    },
    "security_engineer": {
        "direct": ["8097321", "34413f8d-26bf-4bbc-8ade-eb309a0e2245"],
        "adjacent": ["8350419", "8168155"],
        "rationale": "Cloudflare and Ramp own hands-on security engineering. Zero-trust and platform-security leadership are adjacent due to manager/director occupational intent.",
    },
    "site_reliability": {
        "direct": [], "adjacent": ["5073998008", "8300518", "be94e89b-89d5-4f2a-a58b-7929c8d97f92"],
        "rationale": "No SRE posting exists; CI reliability, capacity engineering, and cloud-native search infrastructure are neighboring operational roles.",
    },
    "software_intern": {
        "direct": ["6219995004", "3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "8386080"], "adjacent": [],
        "rationale": "Only postings explicitly offering software-engineering internships are relevant; applied-science and data-science internships are different occupations.",
    },
    "new_grad_engineer": {
        "direct": ["8072244"], "adjacent": ["8242623", "8384186"],
        "rationale": "Roblox Software Engineer is explicitly early career; specialized PhD early-career ML/data roles are adjacent.",
    },
    "india_backend_only": {
        "direct": ["8401112", "8105716"], "adjacent": [],
        "forbidden": ["5073998008", "6233040004", "30a5c739-185f-4a55-8a63-85974d687444", "3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c", "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb", "a753a9e7-a361-426b-9c25-3cf2488c1730", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184", "34413f8d-26bf-4bbc-8ade-eb309a0e2245", "7bfa613e-151c-469b-9973-c89ee3d14838", "8072244"],
        "unresolved": {"8401112": "Board location is only 'Hybrid'; India eligibility is not stated.", "8105716": "Board location is only 'Hybrid'; India eligibility is not stated."},
        "rationale": "Relevant roles with confirmed non-India locations are forbidden. Cloudflare roles retain relevance but have unresolved country eligibility rather than assumed ineligibility.",
    },
    "remote_support_only": {
        "direct": ["00c2c97b-8514-4617-9883-e53e486b6dcd", "4cba9c95-d16f-440d-83e7-2352480f689f"],
        "adjacent": ["0dbd1790-2652-49fc-861f-7451ae4eb1cf", "d8890a6d-ce76-4c3d-aacc-34c1ed150d91", "4967b1cf-a1f7-417b-b650-feebbff767a3", "6b2da750-51fb-4428-b682-fe2296d9aef7", "6104028004"],
        "forbidden": ["5424443008", "8301102", "6232584004", "ff294bf0-7dc1-48d6-876d-25acc8de264e", "8286666"],
        "unresolved": {"6104028004": "The posting location is São Paulo but the work policy is not stated."},
        "rationale": "Remote technical support/enablement is direct; remote solutions/customer-success roles are adjacent. Hybrid/onsite related roles are forbidden and unknown policy remains unresolved.",
    },
    "junior_max_two_years": {
        "direct": ["8072244"],
        "adjacent": ["6219995004", "3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "8386080", "8401112", "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184", "8242623"],
        "forbidden": ["5073998008", "8105716", "34413f8d-26bf-4bbc-8ade-eb309a0e2245", "7bfa613e-151c-469b-9973-c89ee3d14838", "8204785"],
        "unresolved": {
            "8401112": "No numeric experience requirement is stated in structured data.",
            "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2": "No numeric experience requirement is stated.",
            "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb": "No numeric experience requirement is stated.",
            "94ccf41e-d3e1-41aa-9569-c3bcbffc4184": "No numeric experience requirement is stated."
        },
        "rationale": "Only related roles with explicit senior/staff/principal signals or over-two-year requirements are forbidden. Unknown experience stays unresolved.",
    },
    "product_designer": {
        "direct": ["8233482"], "adjacent": [],
        "rationale": "Roblox is product design. Brand design is a different occupational family despite sharing design terminology.",
    },
    "technical_recruiter": {
        "direct": [], "adjacent": [],
        "rationale": "The frozen corpus contains no recruiting or talent-acquisition role.",
    },
    "enterprise_sales": {
        "direct": ["5426617008", "6dd5500a-4bd4-4f57-ba09-7b07acefecf4"],
        "adjacent": ["7695702", "8130580", "6243897004", "cb8e77a6-4316-45c6-ac8d-7ade980cd65e", "7ee0355b-85dd-4019-8e7f-8f3f38dc17cb", "5551fd7b-9ede-460e-9b38-4328c476a521", "f7c4de3c-1d64-42e9-8111-f38e265327d8"],
        "rationale": "Enterprise account-executive ownership is direct; adjacent account segments and technical sales are useful neighboring sales roles.",
    },
    "legal_counsel": {
        "direct": [], "adjacent": [],
        "rationale": "AI strategy for legal customers is not practicing commercial counsel; no counsel posting exists in the frozen sample.",
    },
    "ambiguous_technical": {
        "direct": [],
        "adjacent": ["5073998008", "5468939008", "5383714008", "8097321", "8300518", "8401112", "8402154", "8105716", "8130580", "8378768", "8350419", "8261861", "5707966004", "6238480004", "6233040004", "6181922004", "6219995004", "6253961004", "62a4524c-75da-499e-b150-7f68ce9f2da3", "d8890a6d-ce76-4c3d-aacc-34c1ed150d91", "30a5c739-185f-4a55-8a63-85974d687444", "ff294bf0-7dc1-48d6-876d-25acc8de264e", "e9bf7ffa-9cf2-4360-a2e4-e9fa66859927", "3979a6f5-2abb-447b-bd3a-0565b06bae24", "4967b1cf-a1f7-417b-b650-feebbff767a3", "3fba1c39-c5cb-47d7-9ad2-1cec4d7e9d0c", "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb", "ff1029bd-bb6d-4d78-a03e-5f9744d0b798", "00c2c97b-8514-4617-9883-e53e486b6dcd", "4cba9c95-d16f-440d-83e7-2352480f689f", "74aed644-d963-4c5d-b7a0-149bc35f9c50", "a753a9e7-a361-426b-9c25-3cf2488c1730", "e916f0c8-5a31-4bb6-b4e3-7c4a64b95fda", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184", "4b5e02a8-6a94-494d-9d49-037f2cfc144d", "34413f8d-26bf-4bbc-8ade-eb309a0e2245", "8efd3a0a-fc66-46e2-9415-bffba10e2919", "b39ceb08-a0a7-4f8b-a760-2fb88e209956", "7bfa613e-151c-469b-9973-c89ee3d14838", "e577622f-6657-4e53-8941-b3a774b04448", "8242621", "8384186", "8242619", "8242623", "8072244", "8386080", "8286666", "8204785"],
        "rationale": "The request is intentionally underspecified, so substantive engineering, science, technical customer, QA, and technical management roles remain adjacent rather than direct.",
    },
    "preference_change_to_ml": {
        "direct": ["8242623"],
        "adjacent": ["5468939008", "5707966004", "ff1029bd-bb6d-4d78-a03e-5f9744d0b798", "b39ceb08-a0a7-4f8b-a760-2fb88e209956", "8242621", "8384186", "8242619"],
        "forbidden": ["8401112", "8105716", "10dfc8bc-99ad-4ca2-ab76-853cb90a92c2", "ab7e3425-81d5-4705-a7b5-cd60c8a45cdb", "be94e89b-89d5-4f2a-a58b-7929c8d97f92", "94ccf41e-d3e1-41aa-9569-c3bcbffc4184", "7bfa613e-151c-469b-9973-c89ee3d14838"],
        "rationale": "The current ML preference replaces backend. ML roles follow the broad ML audit; substantive backend roles are explicitly negated and forbidden.",
    },
    "no_cobol_mainframe": {"direct": [], "adjacent": [], "rationale": "No posting contains substantive COBOL/mainframe responsibilities."},
    "no_biomedical_lab": {"direct": [], "adjacent": [], "rationale": "No posting is biomedical wet-lab work."},
}


NO_CURRENT_VACANCY_EXTERNAL_IDS = {"4980436008"}


def _responsibility_excerpt(job: Job, limit: int = 520) -> str:
    text = " ".join(job.description.split())
    lowered = text.casefold()
    markers = (
        "note: we don't have open", "about the role", "the role", "responsibilities",
        "you will", "what you'll do", "what you’ll do", "what you'll achieve",
    )
    positions = [lowered.find(marker) for marker in markers if lowered.find(marker) >= 0]
    start = min(positions) if positions else 0
    return text[start:start + limit]


def _resolve_ids(jobs: list[Job]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    duplicates: set[str] = set()
    for job in jobs:
        if job.external_id in resolved:
            duplicates.add(job.external_id)
        resolved[job.external_id] = job.stable_id
    if duplicates:
        raise ValueError(f"external IDs are not unique in holdout: {sorted(duplicates)}")
    return resolved


def _stratified_irrelevant_sample(case: dict, jobs_by_id: dict[str, Job], size: int = 5) -> list[str]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for stable_id, row in case["labels"].items():
        if row["label"] == "irrelevant":
            buckets[jobs_by_id[stable_id].company].append(stable_id)
    ordered_companies = sorted(
        buckets,
        key=lambda company: hashlib.sha256(f"{case['name']}:{company}".encode()).hexdigest(),
    )
    sample: list[str] = []
    for company in ordered_companies:
        candidates = sorted(
            buckets[company],
            key=lambda stable_id: hashlib.sha256(
                f"{case['name']}:{stable_id}".encode()
            ).hexdigest(),
        )
        if candidates:
            sample.append(candidates[0])
        if len(sample) == size:
            break
    return sample


def build_ai_audited_labels(
    corpus_path: str | Path,
    provisional_path: str | Path,
    output_path: str | Path,
    review_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    provisional = json.loads(Path(provisional_path).read_text(encoding="utf-8"))
    if canonical_sha256(corpus["records"]) != corpus["records_sha256"]:
        raise ValueError("holdout corpus checksum mismatch")
    if canonical_sha256(provisional["cases"]) != provisional["labels_sha256"]:
        raise ValueError("provisional labels checksum mismatch")

    jobs = [Job.model_validate(record["job"]) for record in corpus["records"]]
    jobs_by_id = {job.stable_id: job for job in jobs}
    external = _resolve_ids(jobs)
    case_names = {case["name"] for case in provisional["cases"]}
    if case_names != set(SEMANTIC_DECISIONS):
        raise ValueError("semantic audit decisions do not cover every frozen intent")

    reviewed_non_irrelevant: set[tuple[str, str]] = set()
    sampled_irrelevant: set[tuple[str, str]] = set()
    targeted_missed_positive_reviews: set[tuple[str, str]] = set()
    corrections: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    audited_cases = []
    label_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()

    for case in provisional["cases"]:
        decision = SEMANTIC_DECISIONS[case["name"]]
        resolved_sets = {
            label: {external[item] for item in decision.get(label, [])}
            for label in ("direct", "adjacent", "forbidden")
        }
        if any(resolved_sets[a] & resolved_sets[b] for a, b in (("direct", "adjacent"), ("direct", "forbidden"), ("adjacent", "forbidden"))):
            raise ValueError(f"overlapping semantic decisions for {case['name']}")
        unresolved = {
            external[item]: reason for item, reason in decision.get("unresolved", {}).items()
        }
        sample = _stratified_irrelevant_sample(case, jobs_by_id)
        sampled_irrelevant.update((case["name"], stable_id) for stable_id in sample)

        labels: dict[str, dict[str, Any]] = {}
        for stable_id, old_row in case["labels"].items():
            job = jobs_by_id[stable_id]
            old_label = old_row["label"]
            if old_label != "irrelevant":
                reviewed_non_irrelevant.add((case["name"], stable_id))

            if job.external_id in NO_CURRENT_VACANCY_EXTERNAL_IDS:
                new_label = "forbidden"
                reason = "explicit_no_current_vacancy"
                evidence = _responsibility_excerpt(job)
            elif stable_id in resolved_sets["forbidden"]:
                new_label = "forbidden"
                reason = "confirmed_explicit_constraint_violation"
                evidence = _responsibility_excerpt(job)
            elif stable_id in resolved_sets["direct"]:
                new_label = "direct"
                reason = "substantive_responsibility_directly_matches_intent"
                evidence = _responsibility_excerpt(job)
            elif stable_id in resolved_sets["adjacent"]:
                new_label = "adjacent"
                reason = "substantive_neighboring_responsibility"
                evidence = _responsibility_excerpt(job)
            else:
                new_label = "irrelevant"
                reason = "role_or_task_mismatch_after_responsibility_review"
                evidence = _responsibility_excerpt(job)

            if old_label != new_label:
                if new_label == "irrelevant" and old_label == "forbidden":
                    reason = "role_mismatch_or_implicit_preference_is_not_an_explicit_forbidden_constraint"
                elif new_label == "irrelevant":
                    reason = "provisional_evidence_was_boilerplate_optional_incidental_or_a_different_occupation"
                elif old_label == "irrelevant" and new_label in {"direct", "adjacent"}:
                    reason = "missed_substantive_responsibility_in_provisional_labels"
                elif new_label == "forbidden" and job.external_id in NO_CURRENT_VACANCY_EXTERNAL_IDS:
                    reason = "explicit_no_current_vacancy"
                elif new_label == "forbidden":
                    reason = "confirmed_explicit_constraint_violation"
                elif old_label in {"direct", "adjacent"} and new_label in {"direct", "adjacent"}:
                    reason = "semantic_regrade_after_responsibility_and_intent_review"

            is_sample = (case["name"], stable_id) in sampled_irrelevant
            is_targeted_new_positive = (
                old_label == "irrelevant" and new_label in {"direct", "adjacent"}
            )
            if is_targeted_new_positive:
                targeted_missed_positive_reviews.add((case["name"], stable_id))
            was_reviewed = old_label != "irrelevant" or is_sample or is_targeted_new_positive or job.external_id in NO_CURRENT_VACANCY_EXTERNAL_IDS
            row: dict[str, Any] = {"label": new_label}
            if was_reviewed:
                row["audit"] = {
                    "status": "ai_reviewed",
                    "reason": reason,
                    "evidence_snippet": evidence,
                    "provisional_label": old_label,
                    "review_scope": (
                        "all_non_irrelevant" if old_label != "irrelevant"
                        else "stratified_irrelevant_sample" if is_sample
                        else "targeted_missed_positive_review" if is_targeted_new_positive
                        else "global_vacancy_validity_audit"
                    ),
                }
            if stable_id in unresolved:
                row["eligibility"] = "unresolved"
                row["unresolved_reason"] = unresolved[stable_id]
                unresolved_rows.append({
                    "case": case["name"], "stable_id": stable_id,
                    "label": new_label, "reason": unresolved[stable_id],
                })
            labels[stable_id] = row
            label_counts[new_label] += 1
            transition_counts[f"{old_label}->{new_label}"] += 1
            if old_label != new_label:
                corrections.append({
                    "case": case["name"], "stable_id": stable_id,
                    "title": job.title, "from": old_label, "to": new_label,
                    "reason": reason, "evidence_snippet": evidence,
                })

        audited_cases.append({
            "name": case["name"], "family": case["family"], "query": case["query"],
            "saved_preferences": case.get("saved_preferences", {}),
            "audit_rationale": decision["rationale"], "labels": labels,
        })

    expected_non_irrelevant = sum(
        row["label"] != "irrelevant"
        for case in provisional["cases"] for row in case["labels"].values()
    )
    if len(reviewed_non_irrelevant) != expected_non_irrelevant:
        raise ValueError("not every provisional non-irrelevant decision was reviewed")

    payload = {
        "version": "1.1-ai-audited",
        "kind": "boardforge_independent_holdout_labels",
        "status": "frozen_ai_audited_not_human_verified",
        "verification_status": "ai_audited_not_human_verified",
        "warning": "This is an independent AI semantic audit, not human verification.",
        "annotation_method": {
            "judgment_basis": "actual role responsibilities, required qualifications, explicit constraints, and query intent",
            "excluded_as_final_evidence": ["regex matches", "company boilerplate", "optional qualifications", "incidental technology mentions", "ranking predictions", "confidence.py"],
            "scope": {
                "all_provisional_non_irrelevant": len(reviewed_non_irrelevant),
                "stratified_provisional_irrelevant_sample": len(sampled_irrelevant),
                "additional_targeted_missed_positive_reviews": len(targeted_missed_positive_reviews),
                "sample_strategy": "five provisional-irrelevant decisions per query, spread across companies with deterministic hash ordering",
            },
        },
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "corpus_records_sha256": corpus["records_sha256"],
        "provisional_labels_sha256": provisional["labels_sha256"],
        "cases": audited_cases,
        "corrections": corrections,
        "unresolved_cases": unresolved_rows,
        "statistics": {
            "decisions": len(audited_cases) * len(jobs),
            "label_counts": dict(label_counts),
            "correction_count": len(corrections),
            "agreement_count": len(audited_cases) * len(jobs) - len(corrections),
            "agreement_rate": round(1 - len(corrections) / (len(audited_cases) * len(jobs)), 4),
            "transitions": dict(sorted(transition_counts.items())),
            "unresolved_count": len(unresolved_rows),
        },
    }
    payload["labels_sha256"] = canonical_sha256(audited_cases)
    output = Path(output_path)
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite AI-audited dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Holdout v1.1 AI semantic audit",
        "",
        "> **AI-audited, not human-verified.** The original corpus and provisional labels remain unchanged.",
        "",
        f"- Corpus checksum: `{corpus['records_sha256']}`",
        f"- Provisional checksum: `{provisional['labels_sha256']}`",
        f"- AI-audited checksum: `{payload['labels_sha256']}`",
        f"- Reviewed all provisional non-irrelevant decisions: {len(reviewed_non_irrelevant)}",
        f"- Stratified provisional-irrelevant sample: {len(sampled_irrelevant)}",
        f"- Additional targeted missed-positive reviews: {len(targeted_missed_positive_reviews)}",
        f"- Corrections: {len(corrections)}; unresolved eligibility decisions: {len(unresolved_rows)}",
        "",
        "## Corrections",
        "",
    ]
    for item in corrections:
        lines.append(
            f"- `{item['case']}` — **{item['from']} → {item['to']}** — "
            f"`{item['stable_id']}` {item['title']} — {item['reason']}. "
            f"Evidence: {item['evidence_snippet'][:360]}"
        )
    lines.extend(["", "## Unresolved eligibility", ""])
    for item in unresolved_rows:
        lines.append(
            f"- `{item['case']}` / `{item['stable_id']}` ({item['label']}): {item['reason']}"
        )
    review = Path(review_path)
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text("\n".join(lines), encoding="utf-8")
    return {
        "output": str(output), "review": str(review),
        "labels_sha256": payload["labels_sha256"], **payload["statistics"],
        "audit_scope": payload["annotation_method"]["scope"],
    }
