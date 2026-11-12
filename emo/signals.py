"""盘中买卖点信号引擎 (纯逻辑, 不访问网络, 便于测试).

每个 tick 输入: 涨幅榜快照 + 持仓行情 + 涨停/炸板池 + 板块强度 + 实时情绪,
与上一 tick 的状态比较, 产出信号.

买点 (只在情绪非 退潮/冰点 时给出 "买", 否则降级为 "观察"):
  B_SEAL   新封板     : 本 tick 新进入涨停池, 封板分高, 属于最强主线 (已回测: 封板分高组次日溢价正)
  B_RESEAL 炸板回封   : 上 tick 在炸板池, 本 tick 回到涨停池 (分歧转一致)
  W_RUSH   冲板预警   : 距涨停 <2.5%, 5分钟涨幅大, 量比/主力占比高, 属于主线 —— 提前准备打板
  W_DIP    主线低吸   : 主线内涨幅 2~6%, 价格站上均价线, 主力大幅净流入
  W_SECTOR 板块异动   : 板块 ~5 分钟内涨幅 +1% 以上且涨停数增加
卖点 (仅针对持仓):
  S_STOP   止损       : 相对成本亏损 >= stop%
  S_BREAK  炸板       : 上 tick 涨停, 本 tick 打开
  S_FADE   冲高回落   : 从日内最高回落 >= 5% 且跌破均价线
  S_WEAK   弱于预期   : 10:00 前, 非今日买入, 绿盘且低于开盘价 (回测: 隔日开盘卖是最优出场)
  S_FLOW   主力流出   : 10:00 后主力净占比 <= -10%
  S_TAKE   止盈       : 浮盈 >= 15% 后从最高回落 >= 3%
  S_SECTOR 板块走弱   : 持仓所在最强板块从日内高点回落 >= 2%
轮动/带动:
  W_ROTATE 轮动新进   : 板块 15 分钟内从主线外进入前 N, 资金净流入增加且涨幅扩大
  W_ACCEL  资金加速   : 15 分钟主力净流入增量最大的板块 (>=2亿)
  W_LEAD   异动带动   : 个股异动(5分钟急拉/封板)后 6 分钟, 板块涨幅/上涨家数/涨停数跟随 => 先锋
  W_ALONE  孤狼异动   : 个股异动但板块不跟, 该股后续封板买点降级为观察
  S_ROTATE 主线退出   : 持仓所在板块掉出主线 (资金流出/涨幅回落)
市场:
  M_EMO    情绪转弱   : 情绪阶段变为 退潮/冰点, 或封板率跌破 60%
  M_OUT    主线退出   : 前 N 主线板块 15 分钟内掉出前 N
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .emotion import is_excluded, lin
from .rotation import LeadTracker, SectorFlow
from .stocks import CORE_W, board_type, core_features

BAD_PHASES = ("退潮", "冰点")

SIG_NAME = {
    "B_SEAL": "新封板", "B_RESEAL": "炸板回封", "W_RUSH": "冲板预警", "W_DIP": "主线低吸",
    "W_SECTOR": "板块异动", "S_STOP": "止损", "S_BREAK": "炸板", "S_FADE": "冲高回落",
    "S_WEAK": "弱于预期", "S_FLOW": "主力流出", "S_TAKE": "止盈", "S_SECTOR": "板块走弱",
    "M_EMO": "情绪转弱", "W_ROTATE": "轮动新进", "W_ACCEL": "资金加速", "W_LEAD": "异动带动",
    "W_ALONE": "孤狼异动", "S_ROTATE": "主线退出", "M_OUT": "主线退出",
}


@dataclass
class Config:
    top_sectors: int = 6        # 视为主线的板块数
    seal_core_min: float = 65   # 新封板买点的最低封板分
    rush_gap: float = 2.5       # 冲板预警: 距涨停 %
    rush_chg5m: float = 1.5     # 冲板预警: 5分钟涨幅 %
    stop: float = 5.0           # 止损 %
    take: float = 15.0          # 止盈启动 %
    take_dd: float = 3.0        # 止盈回撤 %
    fade_dd: float = 5.0        # 冲高回落 %
    cooldown_min: int = 30      # 同一信号冷却分钟数
    lead_chg5m: float = 3.0     # 异动: 5分钟涨幅 % (20cm/北交所 x1.6)
    lead_scope: int = 12        # 异动监测的板块范围(前N强)


@dataclass
class Signal:
    time: str
    kind: str            # B_/W_/S_/M_ 代码
    action: str          # 买 / 观察 / 卖 / 减仓 / 风险
    code: str
    name: str
    price: float
    reason: str
    score: float = 0.0

    @property
    def label(self) -> str:
        return SIG_NAME.get(self.kind, self.kind)


@dataclass
class State:
    """跨 tick 保存的状态."""
    zt_codes: set = field(default_factory=set)
    zb_codes: set = field(default_factory=set)
    at_limit: set = field(default_factory=set)            # 上 tick 处于涨停价的持仓
    sector_hist: dict = field(default_factory=dict)       # bk -> deque[(ts, pct, zt_cnt)]
    sector_peak: dict = field(default_factory=dict)       # bk -> 日内最高涨幅
    fired: dict = field(default_factory=dict)             # (kind, code) -> 上次触发时间
    phase: str = ""
    seal_rate: float = 1.0
    initialized: bool = False


def limit_price(pre_close: float, code: str, name: str = "") -> float:
    pct = 0.05 if "ST" in str(name) else {"20cm": 0.2, "北交所": 0.3}.get(board_type(code), 0.1)
    # 交易所规则: 四舍五入到分
    return float(int(pre_close * (1 + pct) * 100 + 0.5)) / 100


def vwap(row) -> float:
    """均价 = 成交额 / 成交量(手*100)."""
    v = row.get("vol") or 0
    return row["amount"] / (v * 100) if v else row.get("price", 0)


def minutes(t: datetime) -> int:
    return t.hour * 60 + t.minute


class Engine:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.st = State()
        self.flow = SectorFlow(top_n=self.cfg.top_sectors)
        self.lead = LeadTracker()

    # ---------- 工具 ----------
    def _fire(self, out: list, now: datetime, kind: str, action: str, code: str, name: str,
              price: float, reason: str, score: float = 0.0):
        key = (kind, code)
        last = self.st.fired.get(key)
        if last is not None and (now - last).total_seconds() < self.cfg.cooldown_min * 60:
            return
        self.st.fired[key] = now
        out.append(Signal(now.strftime("%H:%M:%S"), kind, action, code, name,
                          round(float(price), 2), reason, round(float(score), 1)))

    @staticmethod
    def _sector_of(code: str, sectors: pd.DataFrame, members: dict):
        for r in sectors.itertuples():
            if code in members.get(r.bk, ()):
                return r
        return None

    # ---------- 主流程 ----------
    def step(self, now: datetime, snap: pd.DataFrame, zt: pd.DataFrame, zb: pd.DataFrame,
             sectors: pd.DataFrame, members: dict, emo: dict,
             positions: pd.DataFrame | None = None, holds: pd.DataFrame | None = None,
             idx_chg5m: float = 0.0) -> list[Signal]:
        """snap: 涨幅榜快照; holds: 持仓实时行情; positions: code,cost,qty,buy_date."""
        out: list[Signal] = []
        cfg, st = self.cfg, self.st
        phase = emo.get("phase", "混沌")
        ok_mood = phase not in BAD_PHASES
        top = sectors.head(cfg.top_sectors)
        zt_codes = set(zt["c"]) if len(zt) else set()
        zb_codes = set(zb["c"]) if len(zb) else set()

        # ===== 市场情绪 =====
        seal = emo.get("seal_rate", 1.0)
        if st.initialized:
            if phase in BAD_PHASES and st.phase not in BAD_PHASES:
                self._fire(out, now, "M_EMO", "风险", "-", "市场", 0,
                           f"情绪阶段 {st.phase}→{phase}, 停止接力, 持仓逢高减")
            if seal < 0.6 <= st.seal_rate:
                self._fire(out, now, "M_EMO", "风险", "-", "市场", 0,
                           f"封板率跌破60% ({seal:.0%}), 炸板增多, 打板谨慎")
        st.phase, st.seal_rate = phase, seal

        # ===== 板块异动 / 板块峰值 =====
        ts = now.timestamp()
        for r in sectors.itertuples():
            h = st.sector_hist.setdefault(r.bk, deque(maxlen=60))
            h.append((ts, r.pct, r.zt_cnt))
            st.sector_peak[r.bk] = max(st.sector_peak.get(r.bk, r.pct), r.pct)
            old = [x for x in h if ts - x[0] >= 240]  # 约 4~6 分钟前
            if old and r.Index < cfg.top_sectors * 2:
                o = old[-1]
                if r.pct - o[1] >= 1.0 and r.zt_cnt > o[2]:
                    self._fire(out, now, "W_SECTOR", "观察", r.bk, r.name, r.pct,
                               f"5分钟 {o[1]:+.1f}%→{r.pct:+.1f}%, 涨停 {o[2]}→{r.zt_cnt}, 龙头 {r.dragon}",
                               r.score)

        # ===== 盘中轮动 =====
        self.flow.update(ts, sectors)
        out_bks = set()
        for ev in self.flow.events(ts):
            if ev.kind == "IN":
                self._fire(out, now, "W_ROTATE", "观察", ev.bk, ev.name, ev.d_pct,
                           f"排名 {ev.rank0 + 1 if ev.rank0 < 99 else '榜外'}→{ev.rank1 + 1}, "
                           f"15分钟资金{ev.d_net:+.1f}亿 涨幅{ev.d_pct:+.1f}%")
            elif ev.kind == "OUT":
                out_bks.add(ev.bk)
                self._fire(out, now, "M_OUT", "风险", ev.bk, ev.name, ev.d_pct,
                           f"排名 {ev.rank0 + 1}→{ev.rank1 + 1 if ev.rank1 < 99 else '榜外'}, "
                           f"15分钟资金{ev.d_net:+.1f}亿 涨幅{ev.d_pct:+.1f}%")
            elif ev.kind == "ACCEL":
                self._fire(out, now, "W_ACCEL", "观察", ev.bk, ev.name, ev.d_pct,
                           f"15分钟主力净流入+{ev.d_net:.1f}亿 (全场最大), 涨幅{ev.d_pct:+.1f}%")

        # ===== 异动带动: 先评估到期事件, 再登记新异动 =====
        for res in self.lead.evaluate(ts, sectors):
            if res.verdict == "带动板块":
                self._fire(out, now, "W_LEAD", "观察", res.code, res.name, 0,
                           f"{res.trigger}后[{res.sector}] 涨幅{res.d_pct:+.1f}% "
                           f"上涨占比{res.d_up:+.0%} 涨停{res.d_zt:+d}, 带动分{res.score:.0f} => 先锋",
                           res.score)
            elif res.verdict == "孤狼":
                self._fire(out, now, "W_ALONE", "观察", res.code, res.name, 0,
                           f"{res.trigger}后[{res.sector}]未跟随 (涨幅{res.d_pct:+.1f}%), 带动分{res.score:.0f}",
                           res.score)
        scope = sectors.head(cfg.lead_scope)
        if st.initialized:
            for code in zt_codes - st.zt_codes:
                sec = self._sector_of(code, scope, members)
                if sec is not None:
                    nm = zt.loc[zt["c"] == code, "n"].iloc[0]
                    self.lead.trigger(ts, code, nm, sec, "封板")
        if len(snap):
            for r in snap.to_dict("records"):
                th = cfg.lead_chg5m * (1.6 if board_type(r["code"]) != "10cm" else 1.0)
                if (r.get("chg5m") or 0) >= th and not is_excluded(r["name"]):
                    sec = self._sector_of(r["code"], scope, members)
                    if sec is not None:
                        self.lead.trigger(ts, r["code"], r["name"], sec,
                                          f"5分钟+{r['chg5m']:.1f}%")

        # ===== 买点: 新封板 / 回封 =====
        if st.initialized and len(zt):
            feats = core_features(zt).set_index("c")
            for code in zt_codes - st.zt_codes:
                if code not in feats.index:
                    continue
                f = feats.loc[code].copy()
                sec = self._sector_of(code, top, members)
                if sec is None or f["one_word"]:
                    continue
                # 板块效应: 行业(hybk)涨停数 与 所属概念主线涨停数 取高者
                s_sec = lin(sec.zt_cnt, 1, 6)
                if s_sec > f["s_sector"]:
                    f["core"] += (s_sec - f["s_sector"]) * CORE_W["sector"] / sum(CORE_W.values())
                late = minutes(now) >= 14 * 60 + 30
                reseal = code in st.zb_codes
                kind = "B_RESEAL" if reseal else "B_SEAL"
                good = f["core"] >= cfg.seal_core_min or (reseal and f["lbc"] >= 2)
                if not good:
                    continue
                lr = self.lead.stats.get(code)
                alone = lr is not None and lr.verdict == "孤狼"
                action = "买" if ok_mood and not late and not alone else "观察"
                fm = self.lead.first_mover.get(sec.bk)
                role = "先锋" if fm and fm[0] == code else ("龙头" if sec.dragon_code == code else "")
                why = (f"{'回封' if reseal else '封板'} {int(f['lbc'])}板 封板分{f['core']:.0f} "
                       f"封单/流通{f['fund_ratio']:.1f}% 炸板{int(f['zbc'])}次 主线[{sec.name}]"
                       + (f" 角色:{role}" if role else ""))
                if alone:
                    why += " (之前异动板块未跟, 孤狼)"
                if late:
                    why += " (尾盘板, 次日溢价低)"
                if not ok_mood:
                    why += f" (情绪{phase}, 只观察)"
                self._fire(out, now, kind, action, code, f["n"], f["p"], why, f["core"])

        # ===== 预警: 冲板 / 主线低吸 =====
        if len(snap):
            for r in snap.to_dict("records"):
                code = r["code"]
                if code in zt_codes or is_excluded(r["name"]) or not r.get("pre_close"):
                    continue
                sec = self._sector_of(code, top, members)
                if sec is None:
                    continue
                lim = limit_price(r["pre_close"], code, r["name"])
                gap = (lim - r["price"]) / r["pre_close"] * 100
                chg5 = r.get("chg5m") or 0
                mr = r.get("main_ratio") or 0
                vr = r.get("vol_ratio") or 0
                if 0 < gap <= cfg.rush_gap and chg5 >= cfg.rush_chg5m and vr >= 1.5 and mr >= 5:
                    self._fire(out, now, "W_RUSH", "观察", code, r["name"], r["price"],
                               f"距涨停{gap:.1f}% 5分钟+{chg5:.1f}% 量比{vr:.1f} 主力{mr:.0f}% 主线[{sec.name}]",
                               mr)
                elif (ok_mood and 2 <= r["pct"] <= 6 and mr >= 15 and r["price"] > vwap(r)
                      and chg5 > 0 and sec.Index < 3 and minutes(now) < 14 * 60 + 30):
                    self._fire(out, now, "W_DIP", "观察", code, r["name"], r["price"],
                               f"主线[{sec.name}]跟风 涨{r['pct']:.1f}% 主力{mr:.0f}% 站上均价{vwap(r):.2f}",
                               mr)

        # ===== 卖点: 持仓 =====
        if positions is not None and holds is not None and len(positions) and len(holds):
            hq = holds.set_index("code")
            today = now.strftime("%Y%m%d")
            at_limit_now = set()
            for p in positions.to_dict("records"):
                code = str(p["code"])
                if code not in hq.index:
                    continue
                q = hq.loc[code]
                price, name = q["price"], q["name"]
                if not price or price != price:
                    continue
                pnl = (price / p["cost"] - 1) * 100
                lim = limit_price(q["pre_close"], code, name)
                if price >= lim - 0.005:
                    at_limit_now.add(code)
                bought_today = str(p.get("buy_date", "")) == today
                m = minutes(now)
                if pnl <= -cfg.stop:
                    self._fire(out, now, "S_STOP", "卖", code, name, price,
                               f"浮亏{pnl:.1f}% 触发止损-{cfg.stop:.0f}%")
                if code in st.at_limit and code not in at_limit_now:
                    self._fire(out, now, "S_BREAK", "卖", code, name, price,
                               f"涨停打开, 现价{q['pct']:+.1f}%")
                hi = q["high"]
                dd_hi = (hi - price) / q["pre_close"] * 100 if q["pre_close"] else 0
                if dd_hi >= cfg.fade_dd and price < vwap(q):
                    self._fire(out, now, "S_FADE", "减仓", code, name, price,
                               f"从最高回落{dd_hi:.1f}% 且跌破均价{vwap(q):.2f}")
                if not bought_today and m < 600 and q["pct"] < 0 and price < q["open"]:
                    self._fire(out, now, "S_WEAK", "卖", code, name, price,
                               f"开盘后弱于预期: {q['pct']:+.1f}% 低于开盘价{q['open']}")
                mr = q.get("main_ratio")
                if m >= 600 and mr is not None and mr == mr and mr <= -10:
                    self._fire(out, now, "S_FLOW", "减仓", code, name, price,
                               f"主力净占比{mr:.0f}%")
                hi_pnl = (hi / p["cost"] - 1) * 100
                if hi_pnl >= cfg.take and hi_pnl - pnl >= cfg.take_dd:
                    self._fire(out, now, "S_TAKE", "卖", code, name, price,
                               f"最高浮盈{hi_pnl:.1f}% 回落到{pnl:.1f}%")
                sec = self._sector_of(code, sectors, members)
                if sec is not None and sec.bk in out_bks:
                    self._fire(out, now, "S_ROTATE", "减仓", code, name, price,
                               f"所属[{sec.name}]掉出主线, 资金在切换")
                if sec is not None and st.sector_peak.get(sec.bk, sec.pct) - sec.pct >= 2.0:
                    self._fire(out, now, "S_SECTOR", "减仓", code, name, price,
                               f"所属[{sec.name}]从高点{st.sector_peak[sec.bk]:+.1f}%回落到{sec.pct:+.1f}%")
            st.at_limit = at_limit_now

        st.zt_codes, st.zb_codes = zt_codes, zb_codes
        st.initialized = True
        return out
