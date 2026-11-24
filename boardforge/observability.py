from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response


LOGGER = logging.getLogger("boardforge.requests")
HTTP_REQUESTS = Counter(
    "boardforge_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "boardforge_http_request_duration_seconds", "HTTP request latency", ["method", "route"]
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
        route = getattr(request.scope.get("route"), "path", request.url.path)
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
