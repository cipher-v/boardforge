from __future__ import annotations

from pydantic import BaseModel, Field


class FieldMap(BaseModel):
    id: str
    title: str
    location: str | None = None
    description: str | None = None
    apply_url: str | None = None
    employment_type: str | None = None


class ConnectorManifest(BaseModel):
    name: str
    endpoint: str
    method: str = "GET"
    response_path: list[str | int] = Field(default_factory=list)
    fields: FieldMap
    confidence: float
    evidence: list[str] = Field(default_factory=list)
