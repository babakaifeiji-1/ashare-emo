"""板块轮动 / 异动带动 / 标的强弱 测试 (离线)."""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from emo import rotation
from emo.signals import Engine

T0 = datetime(2026, 10, 16, 10, 0, 0)


# ---------------- 日线轮动 ----------------
@pytest.mark.parametrize("counts,status", [
    ([0, 0, 0, 0, 0, 1, 0, 1, 5], "新启动"),
    ([6, 5, 2, 0, 1, 0, 0, 1, 4], "回流"),
    ([1, 1, 3, 4, 5, 5], "主升"),
    ([0, 1, 1, 2, 2, 4], "加强"),
    ([1, 3, 8, 6, 3, 2], "退潮"),
    ([3, 4, 3, 3, 1, 3], "活跃"),
    ([0, 1, 0, 1, 0, 0], ""),
])
def test_rotation_status(counts, status):
    assert rotation.rotation_status(counts) == status


def test_spark():
    assert rotation.spark(0, 10) == "·"
    assert rotation.spark(1, 10) == "▁" and rotation.spark(10, 10) == "█"


class FakeStore:
    def __init__(self, pools):
        self.pools = pools

    def trade_days(self, n):
        return sorted(self.pools)

    def pool(self, kind, d):
        return self.pools[d]


def zt(day_rows):
    return pd.DataFrame([{"c": c, "n": f"股{c}", "lbc": l, "fbt": 93000, "hybk": h}
                         for c, h, l in day_rows])


def test_industry_rotation():
    pools = {}
    for i in range(6):
        rows = [(f"60000{j}", "军工", 1) for j in range(1 if i < 5 else 5)]
        rows += [(f"00000{j}", "汽车", 1 + i) for j in range(4 if i < 3 else 1)]
        pools[f"202610{10 + i}"] = zt(rows)
    r = rotation.industry_rotation(FakeStore(pools), days=6).set_index("industry")
    assert r.loc["军工", "status"] == "新启动" and r.loc["军工", "today"] == 5
    assert r.loc["汽车", "status"] == "退潮"
    assert r.loc["汽车", "lead"].endswith("(6板)")


# ---------------- 盘中轮动 ----------------
def sec(rows):
    """rows: [(bk, pct, main_net, zt_cnt, up)] 已按强度排序."""
    return pd.DataFrame([{"bk": b, "name": f"板块{b}", "pct": p, "main_net": n, "zt_cnt": z,
                          "up": u, "size": 40, "score": 80 - i, "theme": f"板块{b}",
                          "dragon": "", "dragon_code": ""}
                         for i, (b, p, n, z, u) in enumerate(rows)])


def test_sector_flow_in_out_accel():
    f = rotation.SectorFlow(window_s=900, top_n=2)
    t = T0.timestamp()
    f.update(t, sec([("A", 3, 10, 3, 30), ("B", 2.5, 8, 2, 28), ("C", 1, 1, 0, 20)]))
    f.update(t + 960, sec([("C", 3, 6, 3, 35), ("A", 3.2, 12, 3, 31), ("B", 1.5, 5, 2, 20)]))
    ev = {(e.kind, e.bk) for e in f.events(t + 960)}
    assert ("IN", "C") in ev and ("OUT", "B") in ev and ("ACCEL", "C") in ev
    d = f.deltas(t + 960).set_index("bk")
    assert d.loc["C", "d_net"] == 5 and d.loc["C", "rank0"] == 2 and d.loc["C", "rank1"] == 0


def test_sector_flow_needs_window():
    f = rotation.SectorFlow(window_s=900)
    t = T0.timestamp()
    f.update(t, sec([("A", 3, 10, 3, 30)]))
    f.update(t + 300, sec([("A", 4, 15, 4, 32)]))
    assert f.deltas(t + 300).empty


# ---------------- 异动带动 ----------------
def test_lead_tracker_leader_and_lone_wolf():
    lt = rotation.LeadTracker(eval_s=360)
    t = T0.timestamp()
    s0 = sec([("A", 1.0, 5, 1, 15), ("B", 1.0, 5, 1, 15)])
    lt.trigger(t, "600001", "先锋股", s0.iloc[0], "5分钟+3.5%")
    lt.trigger(t, "600009", "孤狼股", s0.iloc[1], "5分钟+3.2%")
    lt.trigger(t + 10, "600002", "同板块", s0.iloc[0], "x")   # 同板块间隔内忽略
    assert len(lt.events) == 2
    assert lt.evaluate(t + 100, s0) == []                     # 未到评估时间
    s1 = sec([("A", 2.8, 9, 4, 30), ("B", 1.0, 5, 1, 14)])
    res = {r.code: r for r in lt.evaluate(t + 400, s1)}
    assert res["600001"].verdict == "带动板块" and res["600001"].d_zt == 3
    assert res["600009"].verdict == "孤狼"
    assert lt.first_mover["A"][0] == "600001" and "B" not in lt.first_mover
    assert lt.events == []


# ---------------- 标的强弱 ----------------
def snap_rows():
    def r(code, name, pct, high, low, price, mr, chg5, fmv):
        return {"code": code, "name": name, "pct": pct, "high": high, "low": low,
                "price": price, "main_ratio": mr, "chg5m": chg5, "float_mv": fmv,
                "amount": price * 1e5 * 100 * 0.98, "vol": 1e5}
    return pd.DataFrame([
        r("600001", "龙头", 10.0, 11, 10, 11.0, 30, 0, 5e9),
        r("600002", "大票", 6.0, 10.8, 10, 10.6, 12, 0.5, 5e10),
        r("600003", "补涨", 8.5, 10.9, 10, 10.85, 20, 1, 3e9),
        r("600004", "弱", 4.0, 10.8, 10, 10.4, -5, -0.5, 3e9),
        r("600005", "*ST垃圾", 5.0, 10.5, 10, 10.5, 0, 0, 1e9),
    ])


def test_strength_roles():
    row = sec([("A", 5.0, 10, 2, 30)]).iloc[0]
    members = {"600001", "600002", "600003", "600004", "600005"}
    z = pd.DataFrame([{"c": "600001", "n": "龙头", "lbc": 3, "fbt": 93000}])
    s = rotation.strength(snap_rows(), row, members, z, idx_chg5m=-0.3,
                          prior={"600003": 1.0, "600004": 20.0})
    roles = dict(zip(s["code"], s["role"]))
    assert "600005" not in roles                                 # ST 剔除
    assert roles["600001"] == "龙头" and roles["600002"] == "中军"
    assert roles["600003"] == "补涨" and roles["600004"] == "跟风"
    assert s.iloc[0]["code"] == "600001"
    assert s.set_index("code").loc["600004", "rs"] == pytest.approx(-1.0)
    s2 = rotation.strength(snap_rows(), row, members, z, first_mover="600003")
    assert dict(zip(s2["code"], s2["role"]))["600003"] == "先锋"


# ---------------- 引擎集成 ----------------
def test_engine_lead_and_rotation_signals():
    e = Engine()
    e.cfg.top_sectors = 2
    e.flow.top_n = 2
    mem = {"A": {"600001"}, "B": {"600002"}, "C": {"600003"}}
    empty = pd.DataFrame(columns=["c"])
    rush = pd.DataFrame([{"code": "600003", "name": "先锋", "price": 10.5, "pre_close": 10,
                          "pct": 5, "chg5m": 3.5, "main_ratio": 3, "vol_ratio": 1,
                          "high": 10.5, "low": 10, "open": 10, "vol": 1e5, "amount": 1e8}])
    emo = {"phase": "发酵", "seal_rate": 0.8}
    s0 = sec([("A", 3, 10, 3, 30), ("B", 2.5, 8, 2, 28), ("C", 1, 1, 0, 15)])
    e.step(T0, rush, empty, empty, s0, mem, emo)
    assert len(e.lead.events) == 1
    s1 = sec([("C", 3, 6, 3, 32), ("A", 3.2, 12, 3, 31), ("B", 1.5, 5, 2, 20)])
    sig = e.step(T0 + timedelta(seconds=960), pd.DataFrame(), empty, empty, s1, mem, emo)
    kinds = {x.kind for x in sig}
    assert {"W_LEAD", "W_ROTATE", "M_OUT", "W_ACCEL"} <= kinds
