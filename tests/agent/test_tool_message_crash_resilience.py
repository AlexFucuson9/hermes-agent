"""Tests for _safe_cute_tool_message crash resilience (#61693).

Cosmetic display/label functions must never abort an agent turn.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestSafeCuteToolMessage:
    """Verify that _safe_cute_tool_message catches display errors."""

    def test_passes_through_on_success(self):
        """Normal call returns the underlying implementation's result."""
        from agent.tool_executor import _safe_cute_tool_message

        with patch(
            "agent.tool_executor._get_cute_tool_message_impl",
            return_value="fancy label",
        ):
            result = _safe_cute_tool_message("web_extract", {"urls": ["https://example.com"]}, 1.5)
        assert result == "fancy label"

    def test_attribute_error_returns_fallback(self):
        """AttributeError (e.g. dict has no .replace) degrades to fallback."""
        from agent.tool_executor import _safe_cute_tool_message

        with patch(
            "agent.tool_executor._get_cute_tool_message_impl",
            side_effect=AttributeError("'dict' object has no attribute 'replace'"),
        ):
            result = _safe_cute_tool_message(
                "web_extract",
                {"urls": [{"url": "https://example.com", "title": "test"}]},
                2.3,
            )
        assert "web_extract" in result
        assert "2.3s" in result

    def test_type_error_returns_fallback(self):
        """TypeError (e.g. regex on dict) degrades to fallback."""
        from agent.tool_executor import _safe_cute_tool_message

        with patch(
            "agent.tool_executor._get_cute_tool_message_impl",
            side_effect=TypeError("expected string or bytes-like object, got 'dict'"),
        ):
            result = _safe_cute_tool_message(
                "web_extract",
                {"urls": [{"url": "https://example.com"}]},
                0.5,
            )
        assert "web_extract" in result
        assert "0.5s" in result

    def test_generic_exception_returns_fallback(self):
        """Any exception from the display function is caught."""
        from agent.tool_executor import _safe_cute_tool_message

        with patch(
            "agent.tool_executor._get_cute_tool_message_impl",
            side_effect=RuntimeError("unexpected display failure"),
        ):
            result = _safe_cute_tool_message("some_tool", {"arg": "val"}, 5.0)
        assert "some_tool" in result
        assert "5.0s" in result

    def test_none_result_handled(self):
        """Passing result=None does not crash."""
        from agent.tool_executor import _safe_cute_tool_message

        with patch(
            "agent.tool_executor._get_cute_tool_message_impl",
            return_value="ok",
        ):
            result = _safe_cute_tool_message("test_tool", {}, 1.0, result=None)
        assert result == "ok"
