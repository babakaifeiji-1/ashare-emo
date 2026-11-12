"""盘中信号引擎测试 (离线, 构造 tick 序列)."""
from datetime import datetime, timedelta

import pandas as pd

from emo.signals import Config, Engine, limit_price

T0 = datetime(2026, 10, 16, 10, 0, 0)
GOOD = {"phase": "发酵", "seal_rate": 0.8}


def sectors_df(pct=5.0, zt_cnt=3):
    return pd.DataFrame({"bk": ["BK1"], "name": ["军工"], "theme": ["军工"], "score": [80.0],
                         "pct": [pct], "zt_cnt": [zt_cnt], "dragon": ["龙头A(3板)"],
                         "dragon_code": ["600001"], "up": [30], "size": [40],
                         "main_net": [10.0]})


MEM = {"BK1": {"600001", "600002", "600003", "600004"}}


def zt_row(code, name="测试", lbc=2, fbt=100000, zbc=0, hs=8.0):
    return {"c": code, "m": 1, "n": name, "p": 11.0, "zdp": 10.0, "amount": 5e8,
            "ltsz": 5e9, "hs": hs, "lbc": lbc, "fbt": fbt, "lbt": fbt, "fund": 2e8,
            "zbc": zbc, "hybk": "军工"}


def zt_df(*rows):
    return pd.DataFrame(list(rows)) if rows else pd.DataFrame(columns=["c"])


def snap_row(code, price, pre=10.0, chg5m=0.0, mr=0.0, vr=1.0, high=None, open_=None,
             vol=1e5, amount=None, name="测试"):
    return {"code": code, "name": name, "price": price, "pre_close": pre,
            "pct": (price / pre - 1) * 100, "chg5m": chg5m, "main_ratio": mr,
            "vol_ratio": vr, "high": high or price, "low": price, "open": open_ or pre,
            "vol": vol, "amount": amount if amount is not None else price * vol * 100 * 0.99}


def step(e, t, snap=(), zt=None, zb=None, sec=None, emo=GOOD, pos=None, holds=None):
    return e.step(t, pd.DataFrame(list(snap)), zt if zt is not None else zt_df(),
                  zb if zb is not None else zt_df(), sec if sec is not None else sectors_df(),
                  MEM, emo, pos, holds)


def kinds(sigs):
    return [s.kind for s in sigs]


def test_limit_price():
    assert limit_price(10.0, "600001") == 11.0
    assert limit_price(10.0, "300001") == 12.0
    assert limit_price(10.0, "600001", "*ST测试") == 10.5
    assert limit_price(3.33, "000001") == 3.66


def test_new_seal_buy_only_after_baseline():
    e = Engine()
    assert "B_SEAL" not in kinds(step(e, T0, zt=zt_df(zt_row("600002"))))  # 首轮=基线
    e2 = Engine()
    step(e2, T0)
    s = step(e2, T0 + timedelta(seconds=20), zt=zt_df(zt_row("600002")))
    b = [x for x in s if x.kind == "B_SEAL"]
    assert len(b) == 1 and b[0].action == "买" and b[0].code == "600002"


def test_seal_outside_main_theme_or_one_word_ignored():
    e = Engine()
    step(e, T0)
    s = step(e, T0 + timedelta(seconds=20),
             zt=zt_df(zt_row("600999"), zt_row("600003", fbt=92500, hs=0.5)))
    assert "B_SEAL" not in kinds(s)


def test_seal_in_bad_mood_is_watch_only():
    e = Engine()
    step(e, T0, emo={"phase": "退潮", "seal_rate": 0.8})
    s = step(e, T0 + timedelta(seconds=20), zt=zt_df(zt_row("600002")),
             emo={"phase": "退潮", "seal_rate": 0.8})
    b = [x for x in s if x.kind == "B_SEAL"][0]
    assert b.action == "观察" and "退潮" in b.reason


def test_reseal():
    e = Engine()
    step(e, T0, zb=zt_df(zt_row("600002")))
    s = step(e, T0 + timedelta(seconds=20), zt=zt_df(zt_row("600002", zbc=1)))
    assert "B_RESEAL" in kinds(s)


def test_rush_warning():
    e = Engine()
    s = step(e, T0, snap=[snap_row("600002", 10.85, chg5m=2.0, mr=12, vr=2.5)])
    assert "W_RUSH" in kinds(s)
    s = step(e, T0, snap=[snap_row("600003", 10.85, chg5m=0.2, mr=12, vr=2.5)])
    assert "W_RUSH" not in kinds(s)


def test_cooldown():
    e = Engine(Config(cooldown_min=30))
    r = [snap_row("600002", 10.85, chg5m=2.0, mr=12, vr=2.5)]
    assert "W_RUSH" in kinds(step(e, T0, snap=r))
    assert "W_RUSH" not in kinds(step(e, T0 + timedelta(minutes=5), snap=r))
    assert "W_RUSH" in kinds(step(e, T0 + timedelta(minutes=31), snap=r))


def test_dip_follower():
    e = Engine()
    s = step(e, T0, snap=[snap_row("600004", 10.4, chg5m=0.5, mr=20, amount=10.0 * 1e5 * 100)])
    assert "W_DIP" in kinds(s)


def test_sector_burst():
    e = Engine()
    step(e, T0, sec=sectors_df(2.0, 1))
    s = step(e, T0 + timedelta(minutes=5), sec=sectors_df(3.5, 3))
    assert "W_SECTOR" in kinds(s)


def test_emotion_turn():
    e = Engine()
    step(e, T0)
    s = step(e, T0 + timedelta(seconds=20), emo={"phase": "退潮", "seal_rate": 0.55})
    assert kinds(s).count("M_EMO") == 1  # 冷却合并为一条


POS = pd.DataFrame([{"code": "600002", "cost": 10.0, "qty": 1000, "buy_date": "20261015"}])


def hold(price, **kw):
    return pd.DataFrame([snap_row("600002", price, **kw)])


def test_stop_loss():
    s = step(Engine(), T0, pos=POS, holds=hold(9.4, open_=9.5))
    assert "S_STOP" in kinds(s)


def test_limit_break():
    e = Engine()
    step(e, T0, pos=POS, holds=hold(11.0))
    s = step(e, T0 + timedelta(seconds=20), pos=POS, holds=hold(10.7, high=11.0))
    assert "S_BREAK" in kinds(s)


def test_weak_open_before_10():
    s = step(Engine(), datetime(2026, 10, 16, 9, 40), pos=POS,
             holds=hold(9.85, open_=9.95, high=9.95))
    assert "S_WEAK" in kinds(s)
    # 今天买入的不触发
    pos_today = POS.assign(buy_date="20261016")
    s = step(Engine(), datetime(2026, 10, 16, 9, 40), pos=pos_today,
             holds=hold(9.85, open_=9.95, high=9.95))
    assert "S_WEAK" not in kinds(s)


def test_fade_and_take_profit():
    pos = POS.assign(cost=9.0)
    s = step(Engine(), T0, pos=pos,
             holds=hold(10.4, high=11.0, amount=10.6 * 1e5 * 100))
    k = kinds(s)
    assert "S_FADE" in k and "S_TAKE" in k


def test_main_outflow_after_10():
    s = step(Engine(), datetime(2026, 10, 16, 10, 30), pos=POS, holds=hold(10.2, mr=-15))
    assert "S_FLOW" in kinds(s)


def test_sector_weaken_for_holding():
    e = Engine()
    step(e, T0, sec=sectors_df(5.0), pos=POS, holds=hold(10.5))
    s = step(e, T0 + timedelta(minutes=10), sec=sectors_df(2.5), pos=POS, holds=hold(10.5))
    assert "S_SECTOR" in kinds(s)
