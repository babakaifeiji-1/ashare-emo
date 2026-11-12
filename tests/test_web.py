"""网页看板: 序列化 / 参数校验 / 定时任务 / HTTP 接口 (不联网)."""
import json
import threading
import urllib.error
import urllib.request
from datetime import datetime
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from emo import web


def test_records_nan_to_null():
    df = pd.DataFrame({"a": [1.5, np.nan], "b": ["x", "y"], "c": [1, 2]})
    out = web.records(df, ["a", "b", "missing"])
    assert out == [{"a": 1.5, "b": "x"}, {"a": None, "b": "y"}]
    json.dumps(out, allow_nan=False)
    assert web.records(None) == [] and web.records(pd.DataFrame()) == []


def test_clean_num():
    assert web.clean_num(float("nan")) is None
    assert web.clean_num(np.float64(2.5)) == 2.5
    assert web.clean_num("日出东方") == "日出东方"


def test_positions_view_pnl():
    pos = pd.DataFrame([{"code": "600000", "cost": 10.0, "qty": 100, "buy_date": "20261112"},
                        {"code": "000001", "cost": 5.0, "qty": 200, "buy_date": "20261111"}])
    holds = pd.DataFrame([{"code": "600000", "name": "浦发银行", "price": 11.0, "pct": 2.0, "main_ratio": 5.0}])
    v = web.positions_view(pos, holds)
    assert v[0]["pnl"] == pytest.approx(10.0) and v[0]["name"] == "浦发银行"
    assert "pnl" not in v[1]                       # 无行情时不算盈亏
    assert web.positions_view(pos, None)[0]["code"] == "600000"


@pytest.mark.parametrize("body", [
    {"op": "add", "code": "60000", "cost": 1, "qty": 1},           # 5 位
    {"op": "add", "code": "../../x", "cost": 1, "qty": 1},
    {"op": "add", "code": "600000", "cost": -1, "qty": 100},
    {"op": "add", "code": "600000", "cost": 10, "qty": 0},
    {"op": "add", "code": "600000", "cost": 10, "qty": 100, "date": "2026-11-12"},
    {"op": "drop", "code": "600000"},
])
def test_parse_pos_rejects(body):
    with pytest.raises((ValueError, TypeError)):
        web.parse_pos(body)


def test_parse_pos_ok():
    assert web.parse_pos({"op": "add", "code": "600000", "cost": "10.5", "qty": "100"}) == \
        ("add", "600000", 10.5, 100, None)
    assert web.parse_pos({"op": "rm", "code": "600000"})[:2] == ("rm", "600000")


def test_due_jobs():
    d = datetime(2026, 11, 12, 15, 10)
    assert web.due_jobs(d, set(), True) == ["report_20261112"]
    assert web.due_jobs(d.replace(minute=40), set(), True) == ["report_20261112", "collect_20261112"]
    assert web.due_jobs(d.replace(minute=40), {"report_20261112", "collect_20261112"}, True) == []
    assert web.due_jobs(d.replace(hour=14), set(), True) == []
    assert web.due_jobs(d, set(), False) == []       # 非交易日不跑


@pytest.fixture
def server(tmp_path, monkeypatch):
    import emo.monitor as mon
    monkeypatch.setattr(mon, "POSITIONS", tmp_path / "positions.csv")
    monkeypatch.setattr(web, "REPORTS", tmp_path)
    (tmp_path / "20261112_1505.md").write_text("# 日报", encoding="utf-8")
    app = web.App()
    app.live = {"emo": {"score": 57.5, "phase": "混沌"}}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app
    srv.shutdown()


def _req(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_endpoints(server):
    base, app = server
    code, body = _req(base + "/")
    assert code == 200 and "A股短线情绪看板".encode() in body
    code, body = _req(base + "/api/state")
    s = json.loads(body)
    assert code == 200 and s["live"]["emo"]["phase"] == "混沌" and s["reports"] == ["20261112_1505.md"]

    assert _req(base + "/api/pos", {"op": "add", "code": "600000", "cost": 10, "qty": 100})[0] == 200
    s = json.loads(_req(base + "/api/state")[1])
    assert [p["code"] for p in s["positions"]] == ["600000"]
    assert _req(base + "/api/pos", {"op": "add", "code": "abc", "cost": 10, "qty": 100})[0] == 400
    assert _req(base + "/api/pos", {"op": "rm", "code": "600000"})[0] == 200
    assert json.loads(_req(base + "/api/state")[1])["positions"] == []

    assert _req(base + "/reports/20261112_1505.md")[1].decode() == "# 日报"
    assert _req(base + "/reports/..%2F..%2Fpositions.csv")[0] == 404   # 路径穿越
    assert _req(base + "/api/job", {"kind": "rm -rf"})[0] == 400
    assert _req(base + "/nope")[0] == 404
