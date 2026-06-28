"""Platform CLI compatibility routes for self-hosted Mem0.

Maps Mem0 Platform API paths (used by @mem0/cli) to the self-hosted REST handlers.
"""

from __future__ import annotations

from typing import Any

from auth import require_admin, verify_auth
from errors import upstream_error
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from schemas import MessageResponse
from server_state import get_memory_instance

router = APIRouter(tags=["cli-compat"])


class CliMessage(BaseModel):
    role: str
    content: str


class CliMemoryAdd(BaseModel):
    messages: list[CliMessage] | None = None
    user_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] | None = None
    infer: bool | None = None
    source: str | None = None


class CliSearch(BaseModel):
    query: str
    filters: dict[str, Any] | None = None
    top_k: int | None = 10
    threshold: float | None = 0.3
    rerank: bool | None = None
    keyword_search: bool | None = None
    source: str | None = None


class CliListMemories(BaseModel):
    filters: dict[str, Any] | None = None
    source: str | None = None


class CliMemoryUpdate(BaseModel):
    text: str | None = None
    metadata: dict[str, Any] | None = None
    source: str | None = None


def _flatten_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if not filters:
        return {}

    if "AND" in filters and isinstance(filters["AND"], list):
        flat: dict[str, Any] = {}
        for clause in filters["AND"]:
            if not isinstance(clause, dict):
                continue
            for key, value in clause.items():
                target = "agent_id" if key == "app_id" else key
                flat[target] = value
        return flat

    if "OR" in filters:
        return {}

    flat = dict(filters)
    if "app_id" in flat and "agent_id" not in flat:
        flat["agent_id"] = flat.pop("app_id")
    return flat


def _scope_from_body(body: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in ("user_id", "agent_id", "app_id", "run_id", "metadata", "infer"):
        value = body.get(key)
        if value is None:
            continue
        if key == "app_id":
            params["agent_id"] = value
        else:
            params[key] = value
    return params


@router.get("/v1/ping/")
def cli_ping(_auth=Depends(verify_auth)):
    return {"status": "ok", "backend": "self-hosted"}


@router.get("/v1/entities/")
def cli_list_entities(_auth=Depends(verify_auth)):
    from routers.entities import list_entities

    return list_entities(_auth=_auth)


@router.delete("/v2/entities/{entity_type}/{entity_id}/")
def cli_delete_entity(entity_type: str, entity_id: str, _auth=Depends(require_admin)):
    from routers.entities import delete_entity

    return delete_entity(entity_type, entity_id, _auth=_auth)


@router.post("/v3/memories/add/")
def cli_add_memory(payload: CliMemoryAdd, _auth=Depends(verify_auth)):
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    params = _scope_from_body(payload.model_dump())
    if not any(params.get(k) for k in ("user_id", "agent_id", "run_id")):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")

    try:
        response = get_memory_instance().add(
            messages=[m.model_dump() for m in payload.messages],
            **{k: v for k, v in params.items() if v is not None},
        )
        return response
    except Exception:
        raise upstream_error()


@router.post("/v3/memories/search/")
def cli_search(payload: CliSearch, _auth=Depends(verify_auth)):
    filters = _flatten_filters(payload.filters)
    params: dict[str, Any] = {}
    if payload.top_k is not None:
        params["top_k"] = payload.top_k
    if payload.threshold is not None:
        params["threshold"] = payload.threshold

    try:
        return get_memory_instance().search(
            query=payload.query,
            filters=filters,
            **params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        raise upstream_error()


@router.post("/v3/memories/")
def cli_list_memories(
    payload: CliListMemories,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    _auth=Depends(verify_auth),
):
    filters = _flatten_filters(payload.filters)
    try:
        if filters:
            result = get_memory_instance().get_all(filters=filters)
            items = result.get("results", result) if isinstance(result, dict) else result
        else:
            results = get_memory_instance().vector_store.list(top_k=page_size)
            rows = results[0] if results and isinstance(results, list) and isinstance(results[0], list) else results or []
            items = []
            for row in rows:
                payload_data = getattr(row, "payload", None) or {}
                items.append(
                    {
                        "id": getattr(row, "id", None),
                        "memory": payload_data.get("data"),
                        "user_id": payload_data.get("user_id"),
                        "agent_id": payload_data.get("agent_id"),
                        "run_id": payload_data.get("run_id"),
                        "created_at": payload_data.get("created_at"),
                        "updated_at": payload_data.get("updated_at"),
                    }
                )

        if not isinstance(items, list):
            items = []
        start = (page - 1) * page_size
        end = start + page_size
        return {"results": items[start:end], "count": len(items)}
    except Exception:
        raise upstream_error()


@router.get("/v1/memories/{memory_id}/")
def cli_get_memory(memory_id: str, _auth=Depends(verify_auth)):
    try:
        return get_memory_instance().get(memory_id)
    except Exception:
        raise upstream_error()


@router.put("/v1/memories/{memory_id}/")
def cli_update_memory(memory_id: str, payload: CliMemoryUpdate, _auth=Depends(verify_auth)):
    if not payload.text:
        raise HTTPException(status_code=400, detail="text is required")
    try:
        return get_memory_instance().update(
            memory_id=memory_id,
            data=payload.text,
            metadata=payload.metadata,
        )
    except Exception:
        raise upstream_error()


@router.delete("/v1/memories/{memory_id}/")
def cli_delete_memory(memory_id: str, _auth=Depends(verify_auth)):
    try:
        get_memory_instance().delete(memory_id=memory_id)
        return MessageResponse(message="Memory deleted successfully")
    except Exception:
        raise upstream_error()


@router.delete("/v1/memories/")
def cli_delete_all_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None,
    run_id: str | None = None,
    _auth=Depends(require_admin),
):
    resolved_agent = agent_id or app_id
    if not any([user_id, resolved_agent, run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v
            for k, v in {
                "user_id": user_id,
                "agent_id": resolved_agent,
                "run_id": run_id,
            }.items()
            if v is not None
        }
        get_memory_instance().delete_all(**params)
        return MessageResponse(message="All relevant memories deleted")
    except Exception:
        raise upstream_error()


@router.get("/v1/events/")
def cli_list_events(_auth=Depends(verify_auth)):
    return {"results": []}


@router.get("/v1/event/{event_id}/")
def cli_get_event(event_id: str, _auth=Depends(verify_auth)):
    return {"id": event_id, "status": "completed", "results": []}
