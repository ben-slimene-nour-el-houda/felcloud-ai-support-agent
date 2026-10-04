"""HTTP endpoints exposing monitoring tools directly to ZeroClaw skills."""
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth.dependencies import get_current_role
from app.tools.monitoring import make_monitoring_tools

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["monitoring"])


class CheckStatusRequest(BaseModel):
    service_name: str

class GetResourceInfoRequest(BaseModel):
    resource_id: str

class GetMetricsRequest(BaseModel):
    service_name: str
    metric_type: str

class GetLogsRequest(BaseModel):
    service_name: str
    lines: int = 100


@router.post("/check_status")
async def check_status_endpoint(body: CheckStatusRequest, role: str = Depends(get_current_role)):
    tools = {t.name: t for t in make_monitoring_tools(role)}
    return tools["check_status"].invoke({"service_name": body.service_name})


@router.post("/get_resource_info")
async def get_resource_info_endpoint(body: GetResourceInfoRequest, role: str = Depends(get_current_role)):
    tools = {t.name: t for t in make_monitoring_tools(role)}
    return tools["get_resource_info"].invoke({"resource_id": body.resource_id})


@router.post("/get_metrics")
async def get_metrics_endpoint(body: GetMetricsRequest, role: str = Depends(get_current_role)):
    tools = {t.name: t for t in make_monitoring_tools(role)}
    return tools["get_metrics"].invoke({"service_name": body.service_name, "metric_type": body.metric_type})


@router.post("/get_logs")
async def get_logs_endpoint(body: GetLogsRequest, role: str = Depends(get_current_role)):
    tools = {t.name: t for t in make_monitoring_tools(role)}
    result = tools["get_logs"].invoke({"service_name": body.service_name, "lines": body.lines})
    return {"logs": result}
