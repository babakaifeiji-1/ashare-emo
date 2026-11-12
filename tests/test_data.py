"""数据层离线测试: 东财K线失败时回退腾讯 (不访问网络)."""
import pytest

from emo.data import KLINE, TX_KLINE, EMClient, Store

TX_ROWS = [  # 腾讯 newfqkline: 日期 开 收 高 低 量(手) {} 换手% 成交额(万元) ...
    ["2026-11-02", "9.98", "10.00", "10.22", "9.93", "1000", {}, "0.30", "100.0", ""],
    ["2026-11-03", "10.06", "11.00", "11.00", "9.85", "2000", {}, "0.40", "250.5", ""],
]


class FakeClient(EMClient):
    def __init__(self, em_ok=False):
        super().__init__()
        self.em_ok, self.calls = em_ok, []

    def get_json(self, url, params, retries=None):
        self.calls.append(url)
        if url == KLINE:
            if not self.em_ok:
                raise RuntimeError("RemoteDisconnected")
            return {"data": {"klines": ["2026-11-03,10,11,11,9.8,2000,2505000,10.0,0.4"]}}
        if url == TX_KLINE:
            sym = params["param"].split(",")[0]
            key = "qfqday" if params["param"].endswith("qfq") else "day"
            return {"data": {sym: {key: TX_ROWS}}}
        raise AssertionError(url)


def test_kline_falls_back_to_tencent_with_same_units():
    c = FakeClient()
    df = c.kline("1.600000", lmt=1, fqt=0)
    assert list(df.columns) == ["date", "open", "close", "high", "low",
                                "vol", "amount", "pct", "turnover"]
    assert len(df) == 1
    r = df.iloc[0]
    assert r["date"] == "20261103"
    assert r["amount"] == pytest.approx(250.5e4)   # 万元 -> 元
    assert r["pct"] == pytest.approx(10.0)          # 由收盘价计算
    assert r["turnover"] == pytest.approx(0.4)


def test_circuit_breaker_skips_eastmoney():
    c = FakeClient()
    c.kline("1.000001", lmt=2)
    c.calls.clear()
    c.kline("0.399001", lmt=2)
    assert c.calls == [TX_KLINE]  # 熔断期间不再请求东财


@pytest.mark.parametrize("secid,sym", [("1.000001", "sh000001"), ("0.399001", "sz399001"),
                                       ("0.920000", "bj920000"), ("0.830799", "bj830799")])
def test_tencent_symbol(secid, sym):
    c = FakeClient()
    seen = []
    orig = c.get_json
    c.get_json = lambda url, params, retries=None: (seen.append(params["param"]), orig(url, params))[1]
    assert len(c.kline_tencent(secid, 2)) == 2
    assert seen[0].startswith(sym + ",day,")


def test_board_kline_not_fallback():
    with pytest.raises(RuntimeError):
        FakeClient().kline("90.BK0480", lmt=4)


def test_eastmoney_used_when_ok():
    df = FakeClient(em_ok=True).kline("1.600000", lmt=1)
    assert df["amount"].iloc[0] == 2505000


def test_index_kline_cached(tmp_path):
    c = FakeClient(em_ok=True)
    s = Store(client=c, cache=tmp_path)
    s.trade_days(1)
    s.market_amount(1)
    assert c.calls.count(KLINE) == 2  # 上证 1 次 + 深成 1 次, 第二次上证走进程内缓存
