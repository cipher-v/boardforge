from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response


LOGGER = logging.getLogger("boardforge.requests")
HTTP_REQUESTS = Counter(
    "boardforge_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "boardforge_http_request_duration_seconds", "HTTP request latency", ["method", "route"]
)
EMBEDDING_REQUESTS = Counter(
    "boardforge_embedding_requests_total",
    "External embedding requests",
    ["provider", "task_type", "status"],
)
EMBEDDING_LATENCY = Histogram(
    "boardforge_embedding_request_duration_seconds",
    "External embedding request latency",
    ["provider", "task_type"],
)
EMBEDDING_BACKLOG = Gauge(
    "boardforge_embedding_backlog_jobs",
    "Jobs awaiting an embedding when maintenance runs",
)


async def request_observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        route_obj = request.scope.get("route")
        route = getattr(route_obj, "path", None) or "__unmatched__"
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
        HTTP_LATENCY.labels(request.method, route).observe(elapsed)
        LOGGER.info(json.dumps({
            "event": "http_request", "request_id": request_id,
            "method": request.method, "route": route,
            "status": status, "duration_ms": round(elapsed * 1000, 2),
        }))


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_embedding_request(
    provider: str,
    task_type: str,
    status: str,
    duration_seconds: float,
) -> None:
    EMBEDDING_REQUESTS.labels(provider, task_type, status).inc()
    EMBEDDING_LATENCY.labels(provider, task_type).observe(duration_seconds)


def observe_embedding_backlog(backlog: int) -> None:
    EMBEDDING_BACKLOG.set(max(0, backlog))
