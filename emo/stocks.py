"""情绪票选股 + 资金打分 + 风控标记.

core_score (只用涨停池字段, 可回测):
    封板时间(越早越强) / 炸板次数 / 封单资金(封单比、封成比) /
    换手(连板需要换手, 首板适中) / 流通市值(情绪票偏好 20~150 亿) /
    行业涨停家数(板块效应) / 连板高度(相对市场最高板)
live 调整 (盘中/收盘快照, 无法用历史复现):
    个股主力净流入占比 + 是否属于当日最强主线 + 主线内地位
"""
from __future__ import annotations

import pandas as pd

from .emotion import clean, lin

CORE_W = {"t_seal": 15, "zbc": 10, "fund": 15, "turnover": 10,
          "mv": 10, "sector": 25, "height": 15}


def trade_minutes(hhmmss) -> float:
    """HHMMSS -> 开盘后第几分钟 (0~240)."""
    try:
        v = int(hhmmss)
    except (TypeError, ValueError):
        return 240.0
    h, m, s = v // 10000, v // 100 % 100, v % 100
    t = h * 60 + m + s / 60
    if t <= 570:
        return 0.0
    if t <= 690:
        return t - 570
    if t <= 780:
        return 120.0
    return min(120 + t - 780, 240.0)


def board_type(code: str) -> str:
    c = str(code)
    if c.startswith(("30", "68")):
        return "20cm"
    if c.startswith(("8", "4", "92")):
        return "北交所"
    return "10cm"


def turnover_score(hs: float, lbc: int) -> float:
    """首板换手 3~15% 最佳; 连板需要充分换手, 8~25% 最佳; 一字/缩量板难买且易被砸."""
    lo, hi = (3, 15) if lbc <= 1 else (8, 25)
    if hs < 1:
        return 10.0
    if hs < lo:
        return 40 + 60 * (hs - 1) / (lo - 1)
    if hs <= hi:
        return 100.0
    return max(0.0, 100 - (hs - hi) * 4)


def mv_score(ltsz: float) -> float:
    y = ltsz / 1e8
    if y < 10:
        return 50.0
    if y <= 150:
        return 100.0
    if y <= 400:
        return 100 - (y - 150) / 250 * 70
    return 30.0


def core_features(zt: pd.DataFrame) -> pd.DataFrame:
    """在某一天涨停池上计算 core_score. 返回带分项的副本."""
    z = clean(zt)
    if z.empty:
        return z
    z = z.copy()
    heat = z.groupby("hybk")["c"].transform("count")
    max_lb = z["lbc"].max()
    z["sector_zt"] = heat
    z["seal_min"] = z["fbt"].map(trade_minutes)
    z["fund_ratio"] = z["fund"] / z["ltsz"] * 100          # 封单/流通市值 %
    z["fund_amt"] = z["fund"] / z["amount"].clip(lower=1)  # 封成比
    z["s_t_seal"] = z["seal_min"].map(lambda m: lin(m, 240, 0))
    z["s_zbc"] = z["zbc"].map(lambda n: {0: 100, 1: 60, 2: 30}.get(int(n), 0))
    z["s_fund"] = (z["fund_ratio"].map(lambda x: lin(x, 0, 1.0))
                   + z["fund_amt"].map(lambda x: lin(x, 0, 1.0))) / 2
    z["s_turnover"] = [turnover_score(h, l) for h, l in zip(z["hs"], z["lbc"])]
    z["s_mv"] = z["ltsz"].map(mv_score)
    z["s_sector"] = heat.map(lambda n: lin(n, 1, 6))
    z["s_height"] = z["lbc"].map(
        lambda l: 100 if (l == max_lb and l >= 3) else 70 if l >= 3 else 55 if l == 2 else 30)
    tot = sum(CORE_W.values())
    z["core"] = sum(z[f"s_{k}"] * w for k, w in CORE_W.items()) / tot
    # 一字板: 开盘即封且换手极低, 散户基本买不到
    z["one_word"] = (z["fbt"].astype(int) <= 92500) & (z["hs"] < 1.5)
    z["btype"] = z["c"].map(board_type)
    return z


def risk_flags(r: dict, phase: str) -> list[str]:
    f = []
    if r.get("one_word"):
        f.append("一字板难买")
    if r.get("lbc", 0) >= 5:
        f.append("高位板")
    if r.get("zbc", 0) >= 3:
        f.append("反复炸板")
    if r.get("seal_min", 0) >= 210:
        f.append("尾盘板")
    if r.get("btype") == "北交所":
        f.append("北交所30cm")
    mr = r.get("main_ratio")
    if mr is not None and mr == mr and mr < 0:
        f.append("主力净流出")
    if phase in ("退潮", "冰点"):
        f.append("情绪" + phase)
    return f


def pick(zt: pd.DataFrame, snap: pd.DataFrame, sectors: pd.DataFrame,
         members: dict, phase: str, top_sectors: int = 8) -> pd.DataFrame:
    """当日情绪票候选.

    zt: 今日涨停池; snap: 全A快照; sectors: rank_sectors 输出; members: 板块成员.
    """
    z = core_features(zt)
    if z.empty:
        return z
    snap = snap.set_index("code")
    z["main_net"] = z["c"].map(snap["main_net"]) / 1e8
    z["main_ratio"] = z["c"].map(snap["main_ratio"])

    # 归属最强板块(主线)
    strong = sectors.head(top_sectors)
    def best_sector(code):
        for r in strong.itertuples():
            if code in members.get(r.bk, ()):
                return r.name, r.theme, r.score, r.dragon_code == code
        return "", "", 0.0, False
    info = z["c"].map(best_sector)
    z["sector"] = info.map(lambda x: x[0])
    z["theme"] = info.map(lambda x: x[1])
    z["sector_score"] = info.map(lambda x: x[2])
    z["is_dragon"] = info.map(lambda x: x[3])

    z["s_flow"] = z["main_ratio"].map(lambda x: lin(x, -10, 20))
    z["s_main"] = z["sector_score"].map(lambda x: lin(x, 40, 85))
    z["score"] = (z["core"] * 0.65 + z["s_flow"] * 0.15 + z["s_main"] * 0.15
                  + z["is_dragon"].astype(float) * 5)
    z["type"] = [
        "龙头" if (d or l >= 3) else ("二板" if l == 2 else "首板")
        for d, l in zip(z["is_dragon"], z["lbc"])
    ]
    z["flags"] = [", ".join(risk_flags(r, phase)) for r in z.to_dict("records")]
    z["score"] = z["score"].round(1)
    return z.sort_values("score", ascending=False).reset_index(drop=True)


def followers(snap: pd.DataFrame, sectors: pd.DataFrame, members: dict,
              zt_codes: set, top_sectors: int = 5, n: int = 15) -> pd.DataFrame:
    """主线内未涨停但资金强势流入的跟风/补涨候选 (适合低吸或次日观察)."""
    rows = []
    s = snap[~snap["name"].astype(str).map(lambda x: "ST" in x or x.startswith(("N", "C")))]
    for r in sectors.head(top_sectors).itertuples():
        m = s[s["code"].isin(members.get(r.bk, ())) & ~s["code"].isin(zt_codes)]
        m = m[(m["pct"] >= 3) & (m["main_net"] > 0)]
        for x in m.itertuples():
            rows.append({"code": x.code, "name": x.name, "sector": r.name, "pct": x.pct,
                         "main_net": x.main_net / 1e8, "main_ratio": x.main_ratio,
                         "turnover": x.turnover, "vol_ratio": x.vol_ratio,
                         "float_mv": x.float_mv / 1e8, "sector_score": r.score})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("code")
    # 资金强度: 主力占比 + 净额 + 量比
    df["score"] = (df["main_ratio"].rank(pct=True) * 40 + df["main_net"].rank(pct=True) * 30
                   + df["vol_ratio"].rank(pct=True) * 15 + df["sector_score"].rank(pct=True) * 15)
    return df.sort_values("score", ascending=False).head(n).round(2).reset_index(drop=True)
