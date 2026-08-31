from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_response_tab_and_operator_acknowledgements_are_present():
    html = (ROOT / "frontend" / "index.html").read_text()
    js = (ROOT / "frontend" / "app.js").read_text()
    assert 'data-detail-tab="response"' in html
    assert 'data-detail-pane="response"' in html
    assert 'id="response-apply-ack"' in html
    assert 'id="response-rollback-ack"' in html
    assert 'id="response-events"' in html
    assert 'id="response-event-filter"' in html
    assert "/api/response/plans" in js
    assert "/api/response/actions/" in js
    assert "X-NIDS-Response-Token" in js


def test_light_theme_tokens_and_no_legacy_dark_alert_backgrounds():
    css = (ROOT / "frontend" / "styles.css").read_text().lower()
    for token in ["--bg: #f4f8fc", "--panel: #ffffff", "--accent: #1565c0", "--text: #17324d"]:
        assert token in css
    for legacy in ["#2a1414", "#2a2114", "#142a1a"]:
        assert legacy not in css
