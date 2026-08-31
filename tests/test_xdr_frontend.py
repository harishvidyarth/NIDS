from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_xdr_campaign_tab_and_panels_are_present():
    html = (ROOT / "frontend" / "index.html").read_text()
    assert 'data-detail-tab="xdr">XDR / CAMPAIGN' in html
    for element_id in (
        "xdr-sensors", "xdr-graph", "xdr-triage-summary", "xdr-playbook",
        "xdr-response-command", "xdr-operator-ack", "xdr-audit", "xdr-hits",
    ):
        assert f'id="{element_id}"' in html
    assert "app.js?v=xdr1" in html and "styles.css?v=xdr1" in html


def test_xdr_frontend_calls_real_routes_and_draws_surprise_edges():
    js = (ROOT / "frontend" / "app.js").read_text()
    for route in ("/api/ingest/zeek", "/api/graph", "/api/triage", "/api/deception/hits",
                  "/api/response/audit", "/api/response/apply", "/api/response/rollback"):
        assert route in js
    assert "xdr-edge-surprise" in js
    css = (ROOT / "frontend" / "styles.css").read_text()
    assert ".xdr-edge-surprise" in css
    assert "stroke: var(--red)" in css
