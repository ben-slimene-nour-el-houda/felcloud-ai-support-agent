import logging
import requests
from typing import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_MONITORING_ROLES = {"admin", "support_agent", "zeroclaw_agent"}


def _check_rbac(caller_role: Optional[str]) -> None:
    """Raises PermissionError if caller_role is not authorized."""
    if caller_role not in ALLOWED_MONITORING_ROLES:
        logger.warning(f"RBAC denied: role '{caller_role}' is not authorized for monitoring tools.")
        raise PermissionError(f"Role '{caller_role}' is not authorized to use monitoring tools.")


class CheckStatusSchema(BaseModel):
    service_name: str = Field(description="Name of the service to check")

class GetResourceInfoSchema(BaseModel):
    resource_id: str = Field(description="Identifier of the resource to inspect")

class GetMetricsSchema(BaseModel):
    service_name: str = Field(description="Name of the service to get metrics for")
    metric_type: str = Field(description="Type of metric (e.g., 'cpu', 'memory', 'latency', 'all')")

class GetLogsSchema(BaseModel):
    service_name: str = Field(description="Name of the service to fetch logs for")
    lines: Optional[int] = Field(default=100, description="Number of tail lines to retrieve")


def _check_status_impl(service_name: str) -> dict:
    logger.info(f"Executing check_status for service: {service_name}")
    if settings.ZEROCLAW_API_URL:
        try:
            headers = {"Authorization": f"Bearer {settings.ZEROCLAW_API_KEY}"} if settings.ZEROCLAW_API_KEY else {}
            response = requests.get(f"{settings.ZEROCLAW_API_URL}/api/v1/status/{service_name}", headers=headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to check status for {service_name}: {e}")
            return {"service": service_name, "status": "unknown", "error": str(e)}

    status_mock = {
        "database": "operational",
        "web_server": "degraded",
        "zeroclaw": "operational"
    }
    status = status_mock.get(service_name.lower(), "unknown")
    return {"service": service_name, "status": status, "message": f"Service {service_name} is {status}"}


def _get_resource_info_impl(resource_id: str) -> dict:
    logger.info(f"Executing get_resource_info for resource: {resource_id}")
    if settings.ZEROCLAW_API_URL:
        try:
            headers = {"Authorization": f"Bearer {settings.ZEROCLAW_API_KEY}"} if settings.ZEROCLAW_API_KEY else {}
            response = requests.get(f"{settings.ZEROCLAW_API_URL}/api/v1/resources/{resource_id}", headers=headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get resource info for {resource_id}: {e}")
            return {"resource_id": resource_id, "status": "unknown", "error": str(e)}

    return {
        "resource_id": resource_id,
        "type": "virtual_machine",
        "ip_address": "192.168.1.100",
        "region": "eu-west-1",
        "state": "running"
    }


def _get_metrics_impl(service_name: str, metric_type: str) -> dict:
    logger.info(f"Executing get_metrics for {service_name}, metric: {metric_type}")
    if settings.ZEROCLAW_API_URL:
        try:
            headers = {"Authorization": f"Bearer {settings.ZEROCLAW_API_KEY}"} if settings.ZEROCLAW_API_KEY else {}
            response = requests.get(f"{settings.ZEROCLAW_API_URL}/api/v1/metrics/{service_name}", params={"type": metric_type}, headers=headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get metrics for {service_name}: {e}")
            return {"service": service_name, "metric": metric_type, "value": "N/A", "error": str(e)}

    metrics = {"cpu": "45%", "memory": "2.1GB", "latency": "120ms"}
    if metric_type.lower() == "all":
        return {"service": service_name, "metrics": metrics}
    val = metrics.get(metric_type.lower(), "N/A")
    return {"service": service_name, "metric": metric_type, "value": val}


def _get_logs_impl(service_name: str, lines: int = 100) -> str:
    logger.info(f"Executing get_logs for {service_name}, lines: {lines}")
    if settings.ZEROCLAW_API_URL:
        try:
            headers = {"Authorization": f"Bearer {settings.ZEROCLAW_API_KEY}"} if settings.ZEROCLAW_API_KEY else {}
            response = requests.get(f"{settings.ZEROCLAW_API_URL}/api/v1/logs/{service_name}", params={"lines": lines}, headers=headers, timeout=5)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to get logs for {service_name}: {e}")
            return f"[ERROR] Could not retrieve logs: {str(e)}"

    return f"[INFO] Service {service_name} started successfully.\n[WARN] High memory usage detected.\n[ERROR] Connection timeout to database."


def make_monitoring_tools(caller_role: str) -> list:
    """
    Build the monitoring toolset bound to a single, verified role.
    caller_role must come from a trusted source (JWT-verified),
    never from user/LLM-controlled input.
    """
    _check_rbac(caller_role)

    def _bound_check_status(service_name: str) -> dict:
        _check_rbac(caller_role)
        return _check_status_impl(service_name)

    def _bound_get_resource_info(resource_id: str) -> dict:
        _check_rbac(caller_role)
        return _get_resource_info_impl(resource_id)

    def _bound_get_metrics(service_name: str, metric_type: str) -> dict:
        _check_rbac(caller_role)
        return _get_metrics_impl(service_name, metric_type)

    def _bound_get_logs(service_name: str, lines: int = 100) -> str:
        _check_rbac(caller_role)
        return _get_logs_impl(service_name, lines)

    return [
        StructuredTool.from_function(
            func=_bound_check_status,
            name="check_status",
            description="Checks the operational status of a specific service.",
            args_schema=CheckStatusSchema,
        ),
        StructuredTool.from_function(
            func=_bound_get_resource_info,
            name="get_resource_info",
            description="Retrieves detailed information about a specific resource or server.",
            args_schema=GetResourceInfoSchema,
        ),
        StructuredTool.from_function(
            func=_bound_get_metrics,
            name="get_metrics",
            description="Fetches performance metrics for a given service.",
            args_schema=GetMetricsSchema,
        ),
        StructuredTool.from_function(
            func=_bound_get_logs,
            name="get_logs",
            description="Retrieves recent log entries for a specified service.",
            args_schema=GetLogsSchema,
        ),
    ]
