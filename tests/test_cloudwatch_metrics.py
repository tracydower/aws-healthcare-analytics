
# Test CloudWatch metric publishing without making actual AWS calls.
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from api.audit import publish_cloudwatch_metrics

EVENT = {
    "total_records": 10,
    "valid_records": 8,
    "invalid_records": 2,
    "warning_count": 3
}
TIMESTAMP = datetime(2026, 11, 21, tzinfo=timezone.utc)

def test_metrics_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CLOUDWATCH_METRICS_ENABLED", raising=False)
    with patch("api.audit.boto3.client") as client:
        publish_cloudwatch_metrics(EVENT, TIMESTAMP)
        client.assert_not_called()

def test_metrics_published_when_enabled(monkeypatch):
    monkeypatch.setenv("CLOUDWATCH_METRICS_ENABLED", "true")
    with patch("api.audit.boto3.client") as client:
        publish_cloudwatch_metrics(EVENT, TIMESTAMP)
        client.assert_called_once_with("cloudwatch")
        kwargs = client.return_value.put_metric_data.call_args.kwargs
        assert kwargs["Namespace"] == "Healthcare/DataQuality"
        actual = {m["MetricName"]: m["Value"] for m in kwargs["MetricData"]}
        assert actual == {
            "ClaimsProcessed": 10,
            "ValidClaims": 8,
            "InvalidClaims": 2,
            "ValidationWarnings": 3
        }

def test_cloudwatch_failure_does_not_interrupt_processing(monkeypatch):
    monkeypatch.setenv("CLOUDWATCH_METRICS_ENABLED", "true")
    with patch("api.audit.boto3.client") as client:
        client.return_value.put_metric_data.side_effect = RuntimeError("AWS unavailable")
        publish_cloudwatch_metrics(EVENT, TIMESTAMP)
        client.return_value.put_metric_data.assert_called_once()
