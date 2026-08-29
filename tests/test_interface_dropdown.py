from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_interface_dropdown_keeps_device_identifier_visible():
    source = (ROOT / "frontend" / "app.js").read_text()
    assert "`${iface.device} — ${description}`" in source
    assert "opt.value = iface.device" in source


def test_interface_dropdown_has_loading_empty_and_error_states():
    source = (ROOT / "frontend" / "app.js").read_text()
    assert "Loading interfaces…" in source
    assert "No capture interfaces found" in source
    assert "Interface API error:" in source
