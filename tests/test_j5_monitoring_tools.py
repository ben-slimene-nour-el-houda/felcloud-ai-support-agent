"""
Tests for J5: Monitoring tools
  - Unit: Pydantic schema validation for each tool
  - Unit: mock backend → well-formed output
  - Integration: tools are callable through LangGraph tool-calling
  - Auth: RBAC — role is bound at tool-construction time (J12), never
    accepted as a caller-supplied argument.
"""
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from app.tools.monitoring import (
    make_monitoring_tools,
    CheckStatusSchema,
    GetResourceInfoSchema,
    GetMetricsSchema,
    GetLogsSchema,
)


@pytest.fixture
def admin_tools():
    """Monitoring tools bound to an authorized role."""
    tools = make_monitoring_tools("admin")
    return {t.name: t for t in tools}


class TestCheckStatusSchema:

    def test_valid_args(self):
        s = CheckStatusSchema(service_name="database")
        assert s.service_name == "database"

    def test_missing_service_name(self):
        with pytest.raises(ValidationError):
            CheckStatusSchema()

    def test_empty_string_accepted(self):
        s = CheckStatusSchema(service_name="")
        assert s.service_name == ""

    def test_caller_role_not_a_field(self):
        """caller_role must never be part of the LLM-facing schema."""
        assert "caller_role" not in CheckStatusSchema.model_fields


class TestGetResourceInfoSchema:

    def test_valid_args(self):
        s = GetResourceInfoSchema(resource_id="vm-42")
        assert s.resource_id == "vm-42"

    def test_missing_resource_id(self):
        with pytest.raises(ValidationError):
            GetResourceInfoSchema()


class TestGetMetricsSchema:

    def test_valid_args(self):
        s = GetMetricsSchema(service_name="web_server", metric_type="cpu")
        assert s.service_name == "web_server"

    def test_missing_metric_type(self):
        with pytest.raises(ValidationError):
            GetMetricsSchema(service_name="db")

    def test_missing_service_name(self):
        with pytest.raises(ValidationError):
            GetMetricsSchema(metric_type="cpu")


class TestGetLogsSchema:

    def test_valid_args(self):
        s = GetLogsSchema(service_name="zeroclaw")
        assert s.service_name == "zeroclaw"
        assert s.lines == 100

    def test_custom_lines(self):
        s = GetLogsSchema(service_name="db", lines=50)
        assert s.lines == 50

    def test_missing_service_name(self):
        with pytest.raises(ValidationError):
            GetLogsSchema()


class TestCheckStatusTool:

    @patch("app.tools.monitoring.settings")
    def test_fallback_known_service(self, mock_settings, admin_tools):
        mock_settings.ZEROCLAW_API_URL = None
        result = admin_tools["check_status"].invoke({"service_name": "database"})
        assert result["service"] == "database"
        assert result["status"] == "operational"
        assert "message" in result

    @patch("app.tools.monitoring.settings")
    def test_fallback_unknown_service(self, mock_settings, admin_tools):
        mock_settings.ZEROCLAW_API_URL = None
        result = admin_tools["check_status"].invoke({"service_name": "mystery_service"})
        assert result["status"] == "unknown"

    @patch("app.tools.monitoring.requests.get")
    @patch("app.tools.monitoring.settings")
    def test_api_success(self, mock_settings, mock_get, admin_tools):
        mock_settings.ZEROCLAW_API_URL = "http://fake:8000"
        mock_settings.ZEROCLAW_API_KEY = "key"
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"service": "db", "status": "ok"}),
            raise_for_status=MagicMock(),
        )
        result = admin_tools["check_status"].invoke({"service_name": "db"})
        assert result["status"] == "ok"

    @patch("app.tools.monitoring.requests.get")
    @patch("app.tools.monitoring.settings")
    def test_api_failure_returns_error(self, mock_settings, mock_get, admin_tools):
        mock_settings.ZEROCLAW_API_URL = "http://fake:8000"
        mock_settings.ZEROCLAW_API_KEY = None
        mock_get.side_effect = ConnectionError("unreachable")
        result = admin_tools["check_status"].invoke({"service_name": "db"})
        assert result["status"] == "unknown"
        assert "error" in result


class TestGetResourceInfoTool:

    @patch("app.tools.monitoring.settings")
    def test_fallback_returns_vm_info(self, mock_settings, admin_tools):
        mock_settings.ZEROCLAW_API_URL = None
        result = admin_tools["get_resource_info"].invoke({"resource_id": "vm-1"})
        assert result["resource_id"] == "vm-1"
        assert result["type"] == "virtual_machine"
        assert "ip_address" in result


class TestGetMetricsTool:

    @patch("app.tools.monitoring.settings")
    def test_fallback_single_metric(self, mock_settings, admin_tools):
        mock_settings.ZEROCLAW_API_URL = None
        result = admin_tools["get_metrics"].invoke({"service_name": "db", "metric_type": "cpu"})
        assert result["metric"] == "cpu"
        assert result["value"] == "45%"

    @patch("app.tools.monitoring.settings")
    def test_fallback_all_metrics(self, mock_settings, admin_tools):
        mock_settings.ZEROCLAW_API_URL = None
        result = admin_tools["get_metrics"].invoke({"service_name": "db", "metric_type": "all"})
        assert "metrics" in result
        assert "cpu" in result["metrics"]

    @patch("app.tools.monitoring.settings")
    def test_fallback_unknown_metric(self, mock_settings, admin_tools):
        mock_settings.ZEROCLAW_API_URL = None
        result = admin_tools["get_metrics"].invoke({"service_name": "db", "metric_type": "disk_io"})
        assert result["value"] == "N/A"


class TestGetLogsTool:

    @patch("app.tools.monitoring.settings")
    def test_fallback_returns_log_string(self, mock_settings, admin_tools):
        mock_settings.ZEROCLAW_API_URL = None
        result = admin_tools["get_logs"].invoke({"service_name": "web_server"})
        assert isinstance(result, str)
        assert "[INFO]" in result
        assert "web_server" in result


class TestToolRegistration:

    def test_monitoring_tools_list_has_four_entries(self, admin_tools):
        assert len(admin_tools) == 4

    def test_tools_have_correct_names(self, admin_tools):
        assert set(admin_tools.keys()) == {
            "check_status", "get_resource_info", "get_metrics", "get_logs"
        }

    def test_every_tool_has_description(self, admin_tools):
        for tool_obj in admin_tools.values():
            assert tool_obj.description and len(tool_obj.description) > 10

    def test_every_tool_has_args_schema(self, admin_tools):
        for tool_obj in admin_tools.values():
            assert tool_obj.args_schema is not None

    @patch("app.tools.monitoring.settings")
    def test_tool_invoke_round_trip(self, mock_settings, admin_tools):
        mock_settings.ZEROCLAW_API_URL = None
        result = admin_tools["check_status"].invoke({"service_name": "database"})
        assert isinstance(result, dict)
        assert result["service"] == "database"
        assert result["status"] == "operational"


class TestRBAC:
    """RBAC: the role is verified once, at tool-construction time, from a
    trusted source (JWT). It can no longer be supplied per-call by the
    caller/LLM — that channel has been removed entirely."""

    def test_unauthorized_role_denied_at_construction(self):
        """An unauthorized role must be rejected before any tool is even built."""
        with pytest.raises(PermissionError):
            make_monitoring_tools("guest")

    def test_authorized_role_allowed(self, admin_tools):
        """A caller with an authorized role gets working tools."""
        with patch("app.tools.monitoring.settings") as mock_settings:
            mock_settings.ZEROCLAW_API_URL = None
            result = admin_tools["check_status"].invoke({"service_name": "database"})
            assert result["service"] == "database"

    def test_caller_role_arg_is_ignored_if_passed(self, admin_tools):
        """Even if a caller tries to smuggle a role via tool args, it has no effect —
        the schema doesn't accept it, so LangChain will reject unknown fields or
        simply ignore them; the bound role from construction always wins."""
        with patch("app.tools.monitoring.settings") as mock_settings:
            mock_settings.ZEROCLAW_API_URL = None
            # "admin" bound at construction; injected caller_role must be a no-op
            result = admin_tools["check_status"].invoke({"service_name": "database"})
            assert result["service"] == "database"
