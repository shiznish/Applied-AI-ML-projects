"""Admin-only read endpoints, gated to the audit:read permission — see
app/auth/models.py's ROLE_PERMISSIONS."""
from fastapi import APIRouter, Depends

from ..audit import audit_log
from ..auth.rbac import require_permission
from ..db.connection import SecurityContext

router = APIRouter()


@router.get("/v1/admin/audit")
async def get_audit(
    limit: int = 50,
    security_context: SecurityContext = Depends(require_permission("audit:read")),
):
    return {"records": audit_log.read_all(limit=limit)}
