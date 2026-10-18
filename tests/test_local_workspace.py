import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from stock_multi_agent.services.local_workspace import LocalWorkspace, WorkspaceError, normalize_ticker


def test_roundtrip_and_non_destructive_remove(tmp_path):
    workspace = LocalWorkspace(tmp_path / "workspace.json")
    assert workspace.list_tickers() == []
    workspace.add_ticker(" fpt ")
    workspace.add_ticker("FPT")
    workspace.save_note("FPT", "Theo dõi tăng trưởng, không phải lệnh mua.")
    second = LocalWorkspace(workspace.path)
    assert second.list_tickers() == ["FPT"]
    assert second.get_note("fpt") == "Theo dõi tăng trưởng, không phải lệnh mua."
    second.remove_ticker("fpt")
    assert workspace.list_tickers() == []
    assert workspace.get_note("FPT")
    second.save_note("FPT", "")
    assert workspace.get_note("FPT") == ""


@pytest.mark.parametrize("value", ["", "../FPT", "FPT<script>", "VCB;rm", "12", "A" * 11, None, 12])
def test_reject_invalid_tickers(value):
    with pytest.raises(WorkspaceError):
        normalize_ticker(value)


@pytest.mark.parametrize("content", ["{broken", "[]", '{"version": 99}', '{"version":1,"tickers":["../FPT"],"notes":{},"preferences":{}}', '{"version":1,"tickers":[],"notes":{},"preferences":{"api_key":"secret"}}'])
def test_corrupt_workspace_never_overwritten(tmp_path, content):
    path = tmp_path / "workspace.json"
    path.write_text(content, encoding="utf-8")
    workspace = LocalWorkspace(path)
    with pytest.raises(WorkspaceError, match="giữ nguyên"):
        workspace.add_ticker("FPT")
    assert path.read_text(encoding="utf-8") == content


def test_preference_and_note_limits(tmp_path):
    workspace = LocalWorkspace(tmp_path / "workspace.json")
    preferences = {"broker_name": "Demo", "broker_url_template": "https://example.com/quote/{ticker}"}
    workspace.save_preferences(preferences)
    assert workspace.get_preferences() == preferences
    result = workspace.get_preferences()
    result["broker_name"] = "changed"
    assert workspace.get_preferences() == preferences
    for invalid in [{"password": "secret"}, {"broker_name": 123}, {"broker_url_template": "javascript:alert(1)"}, {"broker_url_template": "https://user:pass@example.com"}, {"broker_url_template": "https://example.com/{password}"}, {"broker_url_template": "https://example.com/{ticker.__class__}"}]:
        with pytest.raises(WorkspaceError):
            workspace.save_preferences(invalid)
    with pytest.raises(WorkspaceError):
        workspace.save_note("FPT", "x" * 10001)
    with pytest.raises(WorkspaceError):
        workspace.save_note("FPT", None)


def test_configured_runtime_path_stays_in_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_AGENT_RUNTIME_DIR", str(tmp_path))
    assert LocalWorkspace().path == tmp_path / "interaction" / "workspace.json"


def test_concurrent_instances_preserve_updates(tmp_path):
    path = tmp_path / "workspace.json"
    def add(index):
        LocalWorkspace(path).add_ticker(f"T{index:03d}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(add, range(30)))
    assert len(LocalWorkspace(path).list_tickers()) == 30


def test_replace_failure_preserves_old_file_and_cleans_temp(tmp_path, monkeypatch):
    workspace = LocalWorkspace(tmp_path / "workspace.json")
    workspace.add_ticker("FPT")
    def deny_replace(*args):
        raise PermissionError("disk denied")
    monkeypatch.setattr("stock_multi_agent.services.local_workspace.os.replace", deny_replace)
    with pytest.raises(WorkspaceError, match="quyền truy cập"):
        workspace.add_ticker("VCB")
    assert json.loads(workspace.path.read_text(encoding="utf-8"))["tickers"] == ["FPT"]
    assert not list(tmp_path.glob(".workspace-*.tmp"))


def test_invalid_parent_reports_actionable_error(tmp_path):
    parent = tmp_path / "file-not-directory"
    parent.write_text("keep", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="quyền truy cập"):
        LocalWorkspace(parent / "workspace.json").list_tickers()
    assert parent.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("url", [
    "https://example.com/?token=abc",
    "https://example.com/?symbol={ticker}&quantity=100",
    "https://example.com/?ticker={ticker}&side=buy",
    "https://example.com/execute",
    "https://127.0.0.1/quote/{ticker}",
])
def test_broker_policy_rejects_secrets_order_parameters_and_local_ips(tmp_path, url):
    workspace = LocalWorkspace(tmp_path / "workspace.json")
    with pytest.raises(WorkspaceError):
        workspace.save_preferences({"broker_url_template": url})
    assert not workspace.path.exists()
    content = json.dumps({"version": 1, "tickers": [], "notes": {}, "preferences": {"broker_url_template": url}})
    workspace.path.write_text(content, encoding="utf-8")
    with pytest.raises(WorkspaceError, match="giữ nguyên"):
        workspace.get_preferences()
    assert workspace.path.read_text(encoding="utf-8") == content
