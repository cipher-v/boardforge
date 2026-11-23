from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl


RemotePolicy = Literal["onsite", "hybrid", "remote", "unknown"]


class SourceRef(BaseModel):
    kind: str
    key: str
    board_url: str


class Job(BaseModel):
    source_kind: str
    source_key: str
    external_id: str
    title: str
    company: str
    location: str = ""
    country: str = ""
    remote_policy: RemotePolicy = "unknown"
    description: str = ""
    apply_url: str = ""
    employment_type: str = ""
    salary_text: str = ""
    min_experience_years: float | None = None
    max_experience_years: float | None = None
    tags: list[str] = Field(default_factory=list)
    posted_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def stable_id(self) -> str:
        return f"{self.source_kind}:{self.source_key}:{self.external_id}"


class Preferences(BaseModel):
    roles: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=lambda: ["India"])
    remote_ok: bool = True
    onsite_ok: bool = True
    min_salary_lpa: float | None = None
    max_required_experience_years: float | None = None
    paid_only: bool = False
    avoid_terms: list[str] = Field(default_factory=list)
    prefer_terms: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Match(BaseModel):
    job: Job
    score: float
    reasons: list[str]
    risks: list[str]
    hard_violations: list[str]


class SyncSummary(BaseModel):
    source_kind: str
    source_key: str
    fetched: int
    inserted: int
    updated: int
    unchanged: int
    closed: int


class SearchRequest(BaseModel):
    profile_id: str = "default"
    query: str = ""
    top_k: int = 10


class PreferenceUpdate(BaseModel):
    profile_id: str = "default"
    text: str


class FeedbackRequest(BaseModel):
    profile_id: str = "default"
    job_id: str
    verdict: Literal["save", "reject", "applied", "interview", "ignore"]
    note: str = ""
