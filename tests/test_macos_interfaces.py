"""macOS interface-name mapping in backend/capture/capture.py.

`tcpdump -D` on macOS only reports pcap interface flags ("Up, Running,
Connection status unknown", "Up, Running, Disconnected", ...), never a
human name. list_interfaces() overlays the real hardware-port names from
`networksetup -listallhardwareports` so the UI dropdown shows
"en0 — Wi-Fi (...)" instead of "en0 — Up, Running, Disconnected".
"""
import subprocess

from backend.capture import capture as capture_mod


# Real `networksetup -listallhardwareports` output from a macOS host.
NETWORKSETUP_SAMPLE = """
Hardware Port: Ethernet Adapter (en4)
Device: en4
Ethernet Address: fa:5c:d2:77:0d:8d

Hardware Port: USB 10/100/1000 LAN
Device: en7
Ethernet Address: 00:e0:4c:bb:e7:83

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 36:52:4a:f3:9c:40

Hardware Port: Wi-Fi
Device: en0
Ethernet Address: 80:a9:97:48:90:ab

Hardware Port: Thunderbolt 1
Device: en1
Ethernet Address: 36:52:4a:f3:9c:40

VLAN Configurations
===================
"""

# Real `tcpdump -D` output from the same host.
TCPDUMP_D_SAMPLE = """1.ap1 [Up, Running, Wireless, Associated]
2.awdl0 [Up, Running, Wireless, Associated]
3.llw0 [Up, Running, Connection status unknown]
4.utun0 [Up, Running]
12.en7 [Up, Running, Connected]
14.lo0 [Up, Running, Loopback]
25.en0 [Up, Running, Disconnected]
26.gif0 [none]
"""


def test_parse_hardware_ports_maps_device_to_port():
    ports = capture_mod._parse_hardware_ports(NETWORKSETUP_SAMPLE)
    assert ports["en0"] == "Wi-Fi"
    assert ports["en7"] == "USB 10/100/1000 LAN"
    assert ports["bridge0"] == "Thunderbolt Bridge"
    assert ports["en4"] == "Ethernet Adapter (en4)"
    assert ports["en1"] == "Thunderbolt 1"
    # networksetup never lists loopback; it is always seeded.
    assert ports["lo0"] == "Loopback"


def test_parse_hardware_ports_empty_on_garbage():
    assert capture_mod._parse_hardware_ports("") == {"lo0": "Loopback"}
    assert capture_mod._parse_hardware_ports("not\nnetworksetup\noutput") == {
        "lo0": "Loopback"
    }


def test_macos_hardware_ports_never_raises_when_command_missing(monkeypatch):
    def _boom(*_a, **_kw):
        raise FileNotFoundError("networksetup")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert capture_mod._macos_hardware_ports() == {"lo0": "Loopback"}


def test_list_interfaces_overlays_friendly_names_on_macos(monkeypatch):
    monkeypatch.setattr(capture_mod, "IS_WINDOWS", False)
    monkeypatch.setattr(capture_mod, "IS_MACOS", True)
    monkeypatch.setattr(
        capture_mod, "_macos_hardware_ports",
        lambda: capture_mod._parse_hardware_ports(NETWORKSETUP_SAMPLE),
    )

    def _fake_run(cmd, *a, **kw):
        assert cmd == ["tcpdump", "-D"]
        return subprocess.CompletedProcess(cmd, 0, TCPDUMP_D_SAMPLE, "")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    by_device = {i["device"]: i for i in capture_mod.list_interfaces()}
    assert by_device["en0"]["name"] == "Wi-Fi"
    assert by_device["en0"]["description"] == "Wi-Fi (Up, Running, Disconnected)"
    assert by_device["en7"]["description"] == "USB 10/100/1000 LAN (Up, Running, Connected)"
    assert by_device["lo0"]["description"] == "Loopback (Up, Running, Loopback)"
    # No networksetup entry -> raw pcap flag text is kept untouched.
    assert by_device["llw0"]["description"] == "Up, Running, Connection status unknown"
    assert by_device["utun0"]["description"] == "Up, Running"


def test_list_interfaces_overlay_is_macos_gated(monkeypatch):
    monkeypatch.setattr(capture_mod, "IS_WINDOWS", False)
    monkeypatch.setattr(capture_mod, "IS_MACOS", False)

    def _explode():
        raise AssertionError("_macos_hardware_ports must not run off-macOS")

    monkeypatch.setattr(capture_mod, "_macos_hardware_ports", _explode)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 0, TCPDUMP_D_SAMPLE, ""),
    )

    by_device = {i["device"]: i for i in capture_mod.list_interfaces()}
    assert by_device["en0"]["description"] == "Up, Running, Disconnected"
    assert by_device["en0"]["name"] == "Up, Running, Disconnected"
