from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_interface_dropdown_shows_friendly_name_and_keeps_device_value():
    source = (ROOT / "frontend" / "app.js").read_text()
    # Visible label is the friendly adapter name; the device id is the
    # option value and the full detail is the hover title.
    assert "opt.textContent = iface.name || iface.device" in source
    assert "opt.value = iface.device" in source
    assert "opt.title = iface.description" in source


def test_interface_dropdown_has_loading_empty_and_error_states():
    source = (ROOT / "frontend" / "app.js").read_text()
    assert "Loading interfaces…" in source
    assert "No capture interfaces found" in source
    assert "Interface API error:" in source
