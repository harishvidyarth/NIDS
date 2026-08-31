from __future__ import annotations

import json

import pytest

from backend.response.adapters import LinuxNftablesAdapter, MacOSPFAdapter, WindowsNetSecurityAdapter
from backend.response.models import ResponseTarget


TARGET = ResponseTarget("198.51.100.7", "203.0.113.20", "tcp", 443)


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [
        (LinuxNftablesAdapter(), "nids_response"),
        (WindowsNetSecurityAdapter(), "NIDS Response"),
        (MacOSPFAdapter(), "com.nids.response"),
    ],
)
def test_render_is_platform_specific_and_uses_validated_target(adapter, expected):
    rendered = adapter.render("11111111-1111-1111-1111-111111111111", [TARGET], 15)
    assert expected in json.dumps(rendered)
    assert "198.51.100.7" in json.dumps(rendered)
    assert rendered["rollback"]


def test_default_adapters_never_mutate_without_privileged_helper():
    adapter = LinuxNftablesAdapter()
    with pytest.raises(RuntimeError, match="privileged helper"):
        adapter.apply({"action_id": "x", "targets": []})


def test_linux_preview_preserves_victim_protocol_and_port_scope():
    adapter = LinuxNftablesAdapter()
    adapter._nft_path = lambda: "/usr/sbin/nft"
    rendered = json.dumps(adapter.render("11111111-1111-1111-1111-111111111111", [TARGET], 15))
    assert "203.0.113.20" in rendered
    assert "dport" in rendered and "443" in rendered
    assert "namespace_prerequisites" in rendered


def test_macos_scan_requires_enabled_pf_and_integrated_anchor():
    outputs = iter([
        {"returncode": 0, "stdout": "Status: Enabled", "stderr": ""},
        {"returncode": 0, "stdout": "com.nids.response", "stderr": ""},
        {"returncode": 0, "stdout": "", "stderr": ""},
    ])
    scan = MacOSPFAdapter(read_runner=lambda _argv: next(outputs)).scan()
    assert scan["anchor_integration_available"] is True


def test_scan_is_read_only_and_stable_with_injected_runner():
    calls = []

    def runner(argv):
        calls.append(argv)
        return {"returncode": 0, "stdout": '{"nftables": []}', "stderr": ""}

    adapter = LinuxNftablesAdapter(read_runner=runner)
    adapter._nft_path = lambda: "/usr/sbin/nft"
    before = adapter.scan()
    after = adapter.scan()
    assert before["fingerprint"] == after["fingerprint"]
    assert calls and all("add" not in call and "delete" not in call for call in calls)
