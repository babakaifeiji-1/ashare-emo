"""板块轮动 / 异动带动 / 标的强弱.

1. industry_rotation : 日线级别轮动 —— 近 N 日各行业涨停数矩阵, 判断 新启动/主升/高潮/退潮/回流
2. SectorFlow        : 盘中轮动 —— 板块 15 分钟资金/涨幅/排名变化, 发现 资金切换、新进主线、主线退出
3. LeadTracker       : 异动带动 —— 个股异动后 6~15 分钟板块是否跟随 (先锋 vs 孤狼)
4. strength          : 标的强弱 —— 相对板块/大盘强度、分时位置、抗跌、封板次序 -> 角色(龙头/先锋/中军/补涨/跟风)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from .emotion import clean, is_excluded, lin
from .stocks import board_type

# ============================ 1. 日线轮动 ============================


def rotation_status(counts: list[int]) -> str:
    """counts: 某行业近 N 日涨停数(旧->新). 返回轮动状态."""
    if not counts:
        return ""
    t = counts[-1]
    prev3 = counts[-4:-1]
    avg3 = sum(prev3) / len(prev3) if prev3 else 0
    peak = max(counts)
    peak_i = len(counts) - 1 - counts[::-1].index(peak)
    hot_days = 0
    for c in reversed(counts):
        if c >= 3:
            hot_days += 1
        else:
            break
    older = counts[:-4]
    if t >= 3 and avg3 < 1.5:
        # 之前热过、冷却后重新走强 => 回流; 否则 => 新启动
        return "回流" if older and max(older) >= 4 else "新启动"
    if hot_days >= 3 and t >= peak * 0.7:
        return "主升"
    if t >= 3 and t >= avg3 * 1.5:
        return "加强"
    if peak >= 4 and peak_i < len(counts) - 1 and t <= peak * 0.4:
        return "退潮"
    if t >= 3:
        return "活跃"
    return ""


def industry_rotation(store, days: int = 10, min_peak: int = 3) -> pd.DataFrame:
    """近 days 个交易日按行业(东财 hybk)统计涨停数 + 轮动状态 + 当日最高板."""
    rows, dates = {}, []
    lead = {}
    for d in store.trade_days(days + 5)[-days:]:
        zt = clean(store.pool("zt", d))
        if zt.empty:
            continue
        dates.append(d)
        rows[d] = zt.groupby("hybk")["c"].count()
        top = zt.sort_values(["lbc", "fbt"], ascending=[False, True]).groupby("hybk").head(1)
        lead[d] = {r.hybk: f"{r.n}({int(r.lbc)}板)" for r in top.itertuples()}
    if not dates:
        return pd.DataFrame()
    m = pd.DataFrame(rows).fillna(0).astype(int)[dates]          # 行业 x 日期
    m = m[m.max(axis=1) >= min_peak]
    last = dates[-1]
    out = pd.DataFrame({
        "industry": m.index,
        "today": m[last].values,
        "avg3": m[dates[-4:-1]].mean(axis=1).round(1).values if len(dates) >= 4 else 0,
        "peak": m.max(axis=1).values,
        "status": [rotation_status(list(r)) for r in m.values],
        "lead": [lead[last].get(i, "") for i in m.index],
        "trend": ["".join(spark(v, max(r)) for v in r) for r in m.values],
    })
    order = {"新启动": 0, "回流": 1, "加强": 2, "主升": 3, "活跃": 4, "退潮": 5, "": 6}
    out["_o"] = out["status"].map(order)
    return (out.sort_values(["_o", "today"], ascending=[True, False])
            .drop(columns="_o").reset_index(drop=True))


def spark(v: int, vmax: int) -> str:
    """单字符柱状图, 按本行最大值归一; 0 显示为 ·."""
    bars = "▁▂▃▄▅▆▇█"
    return bars[min(7, max(0, round(v / max(vmax, 1) * 8) - 1))] if v else "·"


# ============================ 2. 盘中轮动 ============================


@dataclass
class FlowEvent:
    kind: str      # IN 新进主线 / OUT 主线退出 / ACCEL 资金加速
    bk: str
    name: str
    d_net: float   # 窗口内主力净流入变化(亿)
    d_pct: float
    rank0: int
    rank1: int


class SectorFlow:
    """记录每个板块的 (ts, pct, main_net, up_ratio, zt_cnt, rank) 时间序列."""

    def __init__(self, window_s: int = 900, top_n: int = 6):
        self.window, self.top_n = window_s, top_n
        self.hist: dict[str, deque] = {}
        self.names: dict[str, str] = {}

    def update(self, ts: float, sec: pd.DataFrame):
        for rank, r in enumerate(sec.itertuples()):
            up_ratio = r.up / r.size if getattr(r, "size", 0) else 0
            self.hist.setdefault(r.bk, deque(maxlen=200)).append(
                (ts, r.pct, r.main_net, up_ratio, r.zt_cnt, rank))
            self.names[r.bk] = r.name
        # 本轮不在候选表的板块记一个"掉出"排名
        seen = set(sec["bk"])
        for bk, h in self.hist.items():
            if bk not in seen and h and h[-1][0] < ts:
                p = h[-1]
                h.append((ts, p[1], p[2], p[3], p[4], 99))

    def at(self, bk: str, ago_s: float, now: float):
        """返回约 ago_s 秒之前的样本 (不足则 None)."""
        h = self.hist.get(bk)
        if not h:
            return None
        old = [x for x in h if now - x[0] >= ago_s]
        return old[-1] if old else None

    def deltas(self, now: float, ago_s: float | None = None) -> pd.DataFrame:
        ago_s = ago_s or self.window
        rows = []
        for bk, h in self.hist.items():
            o = self.at(bk, ago_s, now)
            if o is None:
                continue
            c = h[-1]
            rows.append({"bk": bk, "name": self.names[bk], "d_pct": c[1] - o[1],
                         "d_net": c[2] - o[2], "d_up": c[3] - o[3], "d_zt": c[4] - o[4],
                         "rank0": o[5], "rank1": c[5], "pct": c[1], "net": c[2]})
        return pd.DataFrame(rows)

    def events(self, now: float) -> list[FlowEvent]:
        d = self.deltas(now)
        out = []
        for r in d.itertuples() if len(d) else []:
            if r.rank1 < self.top_n <= r.rank0 and r.d_net > 0 and r.d_pct >= 0.5:
                out.append(FlowEvent("IN", r.bk, r.name, r.d_net, r.d_pct, r.rank0, r.rank1))
            elif r.rank0 < self.top_n <= r.rank1 and (r.d_net < 0 or r.d_pct < 0):
                out.append(FlowEvent("OUT", r.bk, r.name, r.d_net, r.d_pct, r.rank0, r.rank1))
        if len(d) >= 3:
            # 资金加速: 15 分钟净流入增量 全部候选里最大, 且显著(>=2亿)
            top = d.nlargest(1, "d_net").iloc[0]
            if top["d_net"] >= 2:
                out.append(FlowEvent("ACCEL", top["bk"], top["name"], top["d_net"],
                                     top["d_pct"], int(top["rank0"]), int(top["rank1"])))
        return out


# ============================ 3. 异动带动 ============================


@dataclass
class LeadEvent:
    t0: float
    code: str
    name: str
    bk: str
    sector: str
    pct0: float
    up0: float
    zt0: int
    trigger: str
    done: bool = False


@dataclass
class LeadResult:
    code: str
    name: str
    sector: str
    bk: str
    trigger: str
    d_pct: float
    d_up: float
    d_zt: int
    score: float

    @property
    def verdict(self) -> str:
        return "带动板块" if self.score >= 60 else ("孤狼" if self.score < 25 else "带动一般")


@dataclass
class LeadTracker:
    """个股异动 -> 观察 eval_s 秒后板块是否跟随."""
    eval_s: int = 360
    expire_s: int = 900
    gap_s: int = 600           # 同一板块两次异动事件最小间隔
    events: list = field(default_factory=list)
    first_mover: dict = field(default_factory=dict)  # bk -> (code, name, score) 当日最佳先锋
    stats: dict = field(default_factory=dict)        # code -> 最好的 LeadResult

    def trigger(self, ts: float, code: str, name: str, sec_row, why: str):
        for e in self.events:
            if e.bk == sec_row.bk and ts - e.t0 < self.gap_s:
                return
        up = sec_row.up / sec_row.size if sec_row.size else 0
        self.events.append(LeadEvent(ts, code, name, sec_row.bk, sec_row.name,
                                     sec_row.pct, up, int(sec_row.zt_cnt), why))

    def evaluate(self, ts: float, sec: pd.DataFrame) -> list[LeadResult]:
        idx = sec.set_index("bk")
        out = []
        for e in self.events:
            if e.done or ts - e.t0 < self.eval_s:
                continue
            e.done = True
            if e.bk not in idx.index:
                continue
            r = idx.loc[e.bk]
            up = r["up"] / r["size"] if r["size"] else 0
            d_pct, d_up, d_zt = r["pct"] - e.pct0, up - e.up0, int(r["zt_cnt"]) - e.zt0
            score = lin(d_pct, 0, 1.5) * 0.5 + lin(d_up, 0, 0.15) * 0.3 + lin(d_zt, 0, 3) * 0.2
            res = LeadResult(e.code, e.name, e.sector, e.bk, e.trigger, d_pct, d_up, d_zt,
                             round(score, 1))
            out.append(res)
            best = self.stats.get(e.code)
            if best is None or res.score > best.score:
                self.stats[e.code] = res
            fm = self.first_mover.get(e.bk)
            if res.score >= 60 and (fm is None or res.score > fm[2]):
                self.first_mover[e.bk] = (e.code, e.name, res.score)
        self.events = [e for e in self.events if not e.done and ts - e.t0 < self.expire_s]
        return out


# ============================ 4. 标的强弱 ============================


def strength(snap: pd.DataFrame, sec_row, members: set, zt: pd.DataFrame,
             idx_chg5m: float = 0.0, prior: dict | None = None,
             first_mover: str | None = None) -> pd.DataFrame:
    """板块内个股强弱 + 角色. snap 为涨幅榜(只含较强个股), prior: code -> 前5日涨幅%."""
    s = snap[snap["code"].isin(members)]
    s = s[~s["name"].map(is_excluded)].copy()
    if s.empty:
        return s
    prior = prior or {}
    z = clean(zt)
    zt_in = z[z["c"].isin(members)].sort_values("fbt") if len(z) else z
    seal_order = {c: i + 1 for i, c in enumerate(zt_in["c"])} if len(zt_in) else {}
    lbc = dict(zip(zt_in["c"], zt_in["lbc"])) if len(zt_in) else {}
    max_lb = max(lbc.values()) if lbc else 0

    rng = (s["high"] - s["low"]).where(lambda x: x > 0)
    s["rs"] = s["pct"] - sec_row.pct                                  # 相对板块
    s["pos"] = ((s["price"] - s["low"]) / rng).fillna(1.0)            # 分时位置 0~1
    s["vwap"] = s["amount"] / (s["vol"] * 100)
    s["above_vwap"] = s["price"] >= s["vwap"]
    s["resist"] = s["chg5m"].fillna(0) - idx_chg5m                   # 相对大盘 5 分钟
    s["seal_no"] = s["code"].map(seal_order)
    s["lbc"] = s["code"].map(lbc).fillna(0).astype(int)
    s["prior5"] = s["code"].map(prior)
    s["float_yi"] = s["float_mv"] / 1e8

    score = (s["rs"].map(lambda x: lin(x, -3, 5)) * 0.25
             + s["pos"] * 100 * 0.15
             + s["above_vwap"].astype(float) * 100 * 0.10
             + s["main_ratio"].fillna(0).map(lambda x: lin(x, -10, 25)) * 0.20
             + s["resist"].map(lambda x: lin(x, -1, 1.5)) * 0.10
             + s["seal_no"].map(lambda n: 0 if n != n else lin(n, 8, 1)) * 0.20)
    s["strength"] = score.round(1)

    dragon = getattr(sec_row, "dragon_code", "")

    def role(r):
        # 优先级: 龙头(最高板/板块龙头) > 先锋(异动带动) > 首封 > 中军(大市值) > 补涨 > 跟风
        if (r.lbc >= 2 and r.lbc == max_lb) or (dragon and r.code == dragon):
            return "龙头"
        if first_mover and r.code == first_mover:
            return "先锋"
        if r.seal_no == 1:
            return "首封"
        if r.float_yi >= 200 and r.pct >= 5 and (r.main_ratio or 0) > 5:
            return "中军"
        lim = {"20cm": 18, "北交所": 25}.get(board_type(r.code), 9)
        p5 = r.prior5
        if p5 == p5 and p5 is not None and p5 <= 5 and r.pct >= lim * 0.7:
            return "补涨"
        if r.seal_no == r.seal_no and r.seal_no is not None and r.seal_no > 3:
            return "跟风"
        if r.strength >= 65:
            return "强势"
        return "跟风" if r.rs < 0 else "活跃"
    s["role"] = [role(r) for r in s.itertuples()]
    s["sector"] = sec_row.name
    cols = ["code", "name", "role", "strength", "pct", "rs", "chg5m", "pos", "above_vwap",
            "main_ratio", "seal_no", "lbc", "prior5", "float_yi", "sector"]
    return s[cols].sort_values("strength", ascending=False).reset_index(drop=True)
