"""Pydantic response models for the web API.

Kept small. Most DB-shaped responses pass through as plain dicts; these
models exist so the OpenAPI doc has a useful shape and so the UI knows
what to expect from POST endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobMarkRequest(BaseModel):
    """Body for POST /api/jobs/{url}/mark."""

    status: str = Field(..., description="One of: applied, failed, reset.")
    reason: str | None = Field(None, description="Optional failure reason for status=failed.")


class StreamHandle(BaseModel):
    """Returned by retailor / reapply POSTs so the UI can subscribe to /api/streams/{id}."""

    stream_id: str
    kind: str  # "tailor" | "apply"
    url: str


class ModelEntry(BaseModel):
    """A single LiteLLM-compatible model option for the picker."""

    id: str
    label: str
    provider: str


class ModelsPayload(BaseModel):
    """Returned by GET /api/models."""

    current: str | None
    options: list[ModelEntry]
    keys: dict[str, str | None]  # provider -> masked preview, or None if not set


class ModelSelectRequest(BaseModel):
    """Body for POST /api/models/select."""

    model_id: str = Field(..., description="LiteLLM model identifier, e.g. gemini/gemini-2.5-flash")


class StatsPayload(BaseModel):
    """Loose wrapper around database.get_stats() for OpenAPI."""

    total: int
    by_site: list[Any]
    score_distribution: list[Any]
    pending_detail: int = 0
    with_description: int = 0
    detail_errors: int = 0
    scored: int = 0
    unscored: int = 0
    tailored: int = 0
    untailored_eligible: int = 0
    tailor_exhausted: int = 0
    with_cover_letter: int = 0
    cover_exhausted: int = 0
    applied: int = 0
    apply_errors: int = 0
    ready_to_apply: int = 0


class CostBucket(BaseModel):
    date: str
    cost_usd: float
    runs: int


class CostPayload(BaseModel):
    total_usd: float
    by_day: list[CostBucket]
