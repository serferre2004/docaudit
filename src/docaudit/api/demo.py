"""Demo endpoints (provided): the synthetic cases, to try the API without writing invoices."""

from fastapi import APIRouter, Depends, HTTPException, status

from docaudit.api.auth import require_api_key
from docaudit.api.schemas import DemoCase
from docaudit.synthetic import generate_agent_cases

router = APIRouter(prefix="/demo", tags=["demo"], dependencies=[Depends(require_api_key)])

_, _CASES = generate_agent_cases(seed=0)
_BY_ID = {c.id: DemoCase(document_id=c.id, expected=c.expected, text=c.document) for c in _CASES}


@router.get("/cases")
async def list_cases() -> list[str]:
    return list(_BY_ID)


@router.get("/cases/{case_id}")
async def get_case(case_id: str) -> DemoCase:
    case = _BY_ID.get(case_id)
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Unknown case {case_id!r}")
    return case
