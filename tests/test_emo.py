"""离线单元测试 (不访问网络)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from emo import backtest, emotion, sectors, stocks
from emo.data import CN_TZ, Store, trading_fraction


def zt_pool(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "c": [f"60{i:04d}" for i in range(n)], "m": 1,
        "n": [f"股{i}" for i in range(n)],
        "p": rng.uniform(5, 30, n).round(2),
        "zdp": 10.0, "amount": rng.uniform(1e8, 1e9, n),
        "ltsz": rng.uniform(2e9, 3e10, n), "hs": rng.uniform(0.5, 25, n),
        "lbc": [3, 2, 2] + [1] * (n - 3),
        "fbt": [92500, 93500, 100000] + [130000 + i * 100 for i in range(n - 3)],
        "fund": rng.uniform(1e7, 3e8, n),
        "zbc": [0, 1, 0] + [i % 4 for i in range(n - 3)],
        "hybk": (["军工"] * 5 + ["汽车"] * n)[:n],
    })


# ---------------- 基础函数 ----------------
def test_lin_bounds_and_inverse():
    assert emotion.lin(5, 0, 10) == 50
    assert emotion.lin(-1, 0, 10) == 0 and emotion.lin(99, 0, 10) == 100
    assert emotion.lin(0, 40, 0) == 100  # 越小越好
    assert emotion.lin(float("nan"), 0, 1) == 50


@pytest.mark.parametrize("t,m", [(92500, 0), (93000, 0), (100000, 30), (113000, 120),
                                 (120000, 120), (130100, 121), (150000, 240)])
def test_trade_minutes(t, m):
    assert stocks.trade_minutes(t) == pytest.approx(m, abs=0.01)


def test_trading_fraction():
    f = lambda h, mi: trading_fraction(datetime(2026, 10, 16, h, mi, tzinfo=CN_TZ))
    assert f(9, 0) == 0 and f(11, 30) == 0.5 and f(12, 30) == 0.5 and f(15, 30) == 1


def test_excluded_names():
    assert emotion.is_excluded("*ST海航") and emotion.is_excluded("N南乳")
    assert not emotion.is_excluded("长城军工")


def test_board_type():
    assert stocks.board_type("300581") == "20cm"
    assert stocks.board_type("920006") == "北交所"
    assert stocks.board_type("601606") == "10cm"


# ---------------- 情绪 ----------------
def test_day_metrics_and_score():
    zt = zt_pool(10)
    zb = zt_pool(3).assign(c=["1", "2", "3"])
    dt = pd.DataFrame({"n": ["a"], "c": ["9"]})
    yzt = pd.DataFrame({"n": ["x", "y", "z", "*ST w"], "c": list("abcd"),
                        "zdp": [10.0, -6.0, 2.0, -5.0], "ylbc": [2, 1, 1, 1]})
    m = emotion.day_metrics(zt, zb, dt, yzt, prev_zt_n=20)
    assert m["zt_n"] == 10 and m["zb_n"] == 3 and m["dt_n"] == 1
    assert m["seal_rate"] == pytest.approx(10 / 13)
    assert m["max_lb"] == 3 and m["lb_n"] == 3
    assert m["promote_rate"] == pytest.approx(3 / 20)
    assert m["yzt_avg"] == pytest.approx(2.0)          # ST 被剔除
    assert m["big_loss_n"] == 1 and m["ylb_avg"] == 10
    s, parts = emotion.score(m)
    assert 0 <= s <= 100 and set(parts) == set(emotion.SCORE_RULES)


def test_score_monotonic():
    base = dict(yzt_avg=0, promote_rate=0.2, zt_n=50, seal_rate=0.7, max_lb=4,
                dt_n=10, big_loss_n=5, amount_ratio=1.0)
    hot = dict(base, yzt_avg=4, promote_rate=0.45, zt_n=120, dt_n=0)
    cold = dict(base, yzt_avg=-3, dt_n=50, big_loss_n=20)
    assert emotion.score(cold)[0] < emotion.score(base)[0] < emotion.score(hot)[0]


@pytest.mark.parametrize("s,s1,s2,ph", [
    (80, 60, 50, "高潮"), (20, 40, 50, "冰点"), (50, 30, 20, "修复"),
    (60, 50, 40, "发酵"), (55, 70, 75, "分歧"), (40, 55, 60, "退潮"),
    (50, 49, 51, "混沌"), (60, 59, 50, "发酵"), (38, 39, 45, "退潮"),
])
def test_phase(s, s1, s2, ph):
    assert emotion.phase(s, s1, s2) == ph


def test_every_phase_has_advice():
    for ph in ["冰点", "退潮", "混沌", "修复", "发酵", "高潮", "分歧"]:
        assert ph in emotion.PHASE_ADVICE


# ---------------- 板块 ----------------
def boards_df():
    return pd.DataFrame({
        "bk": ["BK1", "BK2", "BK3", "BK4", "BK5"],
        "name": ["国防军工", "航空装备Ⅱ", "航空装备Ⅲ", "融资融券", "小板块"],
        "pct": [6.0, 7.0, 7.0, 1.0, 9.0], "amount": [1e10, 5e9, 5e9, 1e11, 1e8],
        "main_net": [2e9, 1e9, 1e9, 5e9, 1e7], "main_ratio": [7, 10, 10, 1, 5],
        "up": [140, 49, 49, 3000, 3], "down": [1, 0, 0, 800, 1], "flat": [0, 0, 0, 81, 0],
    })


def test_prepare_filters_noise_dup_and_small():
    b = sectors.prepare(boards_df())
    assert set(b["name"]) == {"国防军工", "航空装备Ⅱ"}
    assert b.loc[b["name"] == "国防军工", "main_net"].iloc[0] == pytest.approx(20)


def test_member_stats():
    zt = zt_pool(10)
    mem = pd.DataFrame({"code": ["600000", "600001", "600009", "000001"],
                        "name": list("abcd"), "pct": [10, 10, 6, 1]})
    st = sectors.member_stats(mem, zt)
    assert st["zt_cnt"] == 3 and st["max_lb"] == 3
    assert st["dragon_code"] == "600000" and st["strong_ratio"] == 0.75


def test_cluster_themes():
    df = pd.DataFrame({"bk": ["A", "B", "C"], "name": ["军工", "航空", "汽车"]})
    mem = {"A": set("abcdef"), "B": set("abcx"), "C": set("uvwxyz")}
    assert sectors.cluster_themes(df, mem) == ["军工", "军工", "汽车"]


# ---------------- 个股 ----------------
def test_core_features_ordering():
    z = stocks.core_features(zt_pool(10))
    assert z["core"].between(0, 100).all()
    # 3板最早封板+行业最热, 应该排在前面
    assert z.sort_values("core", ascending=False)["c"].iloc[0] in {"600000", "600001", "600002"}
    assert z.loc[z["c"] == "600000", "s_height"].iloc[0] == 100
    assert bool(z.loc[z["c"] == "600000", "one_word"].iloc[0]) == (z.loc[0, "hs"] < 1.5)


def test_turnover_and_mv_score():
    assert stocks.turnover_score(10, 1) == 100 and stocks.turnover_score(0.5, 1) == 10
    assert stocks.turnover_score(15, 3) == 100 and stocks.turnover_score(50, 3) == 0
    assert stocks.mv_score(50e8) == 100 and stocks.mv_score(1000e8) == 30


def test_pick_and_flags():
    zt = zt_pool(10)
    snap = pd.DataFrame({"code": zt["c"], "name": zt["n"], "main_net": 1e8,
                         "main_ratio": [20] * 5 + [-5] * 5})
    sec = pd.DataFrame({"bk": ["BK1"], "name": ["军工"], "theme": ["军工"],
                        "score": [85.0], "dragon_code": ["600000"]})
    mem = {"BK1": set(zt["c"][:5])}
    p = stocks.pick(zt, snap, sec, mem, "退潮")
    assert len(p) == 10 and p["score"].is_monotonic_decreasing
    top = p.iloc[0]
    assert top["c"] == "600000" and top["type"] == "龙头" and top["sector"] == "军工"
    assert all("情绪退潮" in f for f in p["flags"])
    assert "主力净流出" in p.loc[p["c"] == "600009", "flags"].iloc[0]


# ---------------- 回测 ----------------
def synth_samples(n_days=10, per_day=20, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        core = rng.uniform(20, 90, per_day)
        ret = (core - 55) / 10 + rng.normal(0, 1, per_day)  # 构造有效因子
        df = pd.DataFrame({"date": f"202609{d:02d}", "core": core, "emo": 50 + d,
                           "phase": "发酵" if d % 3 else "退潮",
                           "board_ok": True, "n_entry_ok": True,
                           "b_open": ret, "b_close": ret, "b_high": ret + 3,
                           "n_open": -ret, "n_close": -ret})
        for k in stocks.CORE_W:
            df[f"s_{k}"] = core + rng.normal(0, 5, per_day)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_backtest_report_detects_signal():
    r = backtest.report(synth_samples(), "board")
    q = r["by_quintile"]
    assert q["open%"].iloc[-1] > q["open%"].iloc[0]
    assert r["ic"]["core"] > 0.5
    st = r["strategy"]
    assert st.loc["每日top3", "avg%"] > st.loc["全部(基准)", "avg%"]
    assert st.loc["每日top3+情绪过滤", "days"] < st.loc["每日top3", "days"]
    r2 = backtest.report(synth_samples(), "next_open")
    assert r2["ic"]["core"] < 0


def test_backtest_mode_filters_unfillable():
    s = synth_samples(3, 10)
    s.loc[:4, "board_ok"] = False
    assert len(backtest.mode_frame(s, "board")) == len(s) - 5


def test_limit_pct():
    assert backtest.limit_pct("300001") == 20 and backtest.limit_pct("600001") == 10


# ---------------- 缓存 ----------------
class FakeClient:
    calls = 0

    def pool(self, kind, date):
        FakeClient.calls += 1
        return zt_pool(3) if kind == "zt" else pd.DataFrame()


def test_store_caches_final_dates(tmp_path):
    st = Store(FakeClient(), cache=tmp_path)
    a = st.pool("zt", "20200101")
    b = st.pool("zt", "20200101")
    assert FakeClient.calls == 1 and len(a) == len(b) == 3
    assert b["c"].iloc[0] == "600000"  # 代码保留前导0/字符串
    assert st.pool("dt", "20200101").empty and st.pool("dt", "20200101").empty
    assert FakeClient.calls == 2
