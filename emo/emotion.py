"""市场情绪: 指标计算 -> 0~100 评分 -> 情绪周期阶段 -> 仓位建议.

核心思路(短线情绪周期):
  冰点 -> 修复 -> 发酵/主升 -> 高潮 -> 分歧 -> 退潮 -> 冰点
赚钱效应(昨日涨停今日溢价、连板晋级率)比涨停家数更能反映接力资金的真实意愿,
所以权重最高; 亏钱效应(跌停、大面)用来识别退潮.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .data import Store


def is_excluded(name: str) -> bool:
    """ST / 新股 / 退市整理不参与情绪统计和选股."""
    n = str(name)
    return "ST" in n or n.startswith(("N", "C", "退")) or "退" in n


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "n" not in df:
        return pd.DataFrame() if df is None else df
    return df[~df["n"].map(is_excluded)].copy()


def lin(x: float, lo: float, hi: float) -> float:
    """线性映射到 0~100, lo->0, hi->100 (lo>hi 表示越小越好)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 50.0
    v = (x - lo) / (hi - lo) * 100
    return max(0.0, min(100.0, v))


# 指标 -> (lo, hi, 权重)
SCORE_RULES = {
    "yzt_avg":      (-3.0, 4.0, 20),   # 昨日涨停今日平均涨幅(%) 赚钱效应
    "promote_rate": (0.10, 0.45, 15),  # 连板晋级率 = 今日连板数 / 昨日涨停数
    "zt_n":         (20, 120, 15),     # 涨停家数(剔除ST/新股)
    "seal_rate":    (0.50, 0.85, 10),  # 封板率 = 涨停/(涨停+炸板)
    "max_lb":       (2, 8, 10),        # 最高连板
    "dt_n":         (40, 0, 10),       # 跌停家数, 越少越好
    "big_loss_n":   (15, 0, 10),       # 昨日涨停今日跌超5% 家数(大面), 越少越好
    "amount_ratio": (0.80, 1.20, 10),  # 两市成交额 / 前5日均值
}


def day_metrics(zt: pd.DataFrame, zb: pd.DataFrame, dt: pd.DataFrame,
                yzt: pd.DataFrame, prev_zt_n: int | None) -> dict:
    zt, zb, dt, yzt = clean(zt), clean(zb), clean(dt), clean(yzt)
    zt_n, zb_n, dt_n = len(zt), len(zb), len(dt)
    lbc = zt["lbc"] if zt_n else pd.Series(dtype=float)
    m = {
        "zt_n": zt_n, "zb_n": zb_n, "dt_n": dt_n,
        "seal_rate": zt_n / (zt_n + zb_n) if zt_n + zb_n else float("nan"),
        "max_lb": int(lbc.max()) if zt_n else 0,
        "lb_n": int((lbc >= 2).sum()),
        "lb3_n": int((lbc >= 3).sum()),
        "yzt_avg": float(yzt["zdp"].mean()) if len(yzt) else float("nan"),
        "yzt_up_ratio": float((yzt["zdp"] > 0).mean()) if len(yzt) else float("nan"),
        "ylb_avg": float(yzt.loc[yzt["ylbc"] >= 2, "zdp"].mean())
        if len(yzt) and (yzt["ylbc"] >= 2).any() else float("nan"),
        "big_loss_n": int((yzt["zdp"] <= -5).sum()) if len(yzt) else 0,
    }
    m["promote_rate"] = m["lb_n"] / prev_zt_n if prev_zt_n else float("nan")
    # 最高板的题材, 用于观察主线
    if zt_n:
        top = zt.sort_values(["lbc", "fbt"], ascending=[False, True]).iloc[0]
        m["dragon"] = f"{top['n']}({int(top['lbc'])}板,{top['hybk']})"
    else:
        m["dragon"] = ""
    return m


def score(m: dict) -> tuple[float, dict]:
    parts = {k: lin(m.get(k, float("nan")), lo, hi) for k, (lo, hi, _) in SCORE_RULES.items()}
    tot = sum(SCORE_RULES[k][2] for k in parts)
    s = sum(parts[k] * SCORE_RULES[k][2] for k in parts) / tot
    return round(s, 1), parts


PHASE_ADVICE = {
    "冰点":   ("0-1成", "情绪极弱, 空仓等待. 冰点次日若出现修复(溢价转正/晋级率回升)是最佳试错点, 只做新题材首板或穿越龙头"),
    "退潮":   ("0-1成", "亏钱效应扩散, 高位股补跌. 不接力, 不抄高位, 保住本金"),
    "混沌":   ("1-3成", "无明确方向, 轻仓做最强板块的核心票, 快进快出"),
    "修复":   ("3-5成", "冰点后的修复, 资金回流. 重点做修复中率先晋级的龙头和新题材首板"),
    "发酵":   ("5-7成", "主线发酵, 赚钱效应好. 做主线龙头/中军, 首板选主线内封板早的"),
    "高潮":   ("3-5成", "情绪高潮(一致), 次日易分歧. 只持有不追高, 逢高兑现, 不打后排"),
    "分歧":   ("2-4成", "高潮后分歧, 只做最高板/核心龙头的分歧转一致, 后排跟风坚决回避"),
}


def phase(s: float, s1: float | None, s2: float | None = None) -> str:
    """根据今日/昨日/前日情绪分判定周期阶段."""
    if s1 is None or math.isnan(s1):
        return "高潮" if s >= 75 else "冰点" if s <= 25 else "混沌"
    d = s - s1
    if s >= 75:
        return "高潮"
    if s <= 25:
        return "冰点"
    if d >= 8:
        # 从低位上来是修复, 从中位上来是发酵
        return "修复" if s1 < 45 else "发酵"
    if d <= -8:
        return "分歧" if s >= 50 else "退潮"
    # 变化不大, 看绝对位置 + 前日趋势
    trend = s - s2 if s2 is not None and not math.isnan(s2) else d
    if s >= 58:
        return "发酵" if trend >= 0 else "分歧"
    if s < 42:
        return "退潮" if trend <= 0 else "修复"
    return "混沌"


@dataclass
class EmotionDay:
    date: str
    metrics: dict
    score: float
    parts: dict
    phase: str

    @property
    def advice(self):
        return PHASE_ADVICE[self.phase]


def history(store: Store, n_days: int = 40) -> pd.DataFrame:
    """回填可得的历史情绪(涨停池历史约20个交易日)."""
    days = store.trade_days(n_days)
    amt = store.market_amount(n_days + 6).set_index("date")
    amt["amount_ratio"] = amt["amount"] / amt["amount"].rolling(5).mean().shift(1)

    rows, prev_zt_n = [], None
    for d in days:
        zt = store.pool("zt", d)
        if zt.empty:  # 超出东财历史范围
            prev_zt_n = None
            continue
        m = day_metrics(zt, store.pool("zb", d), store.pool("dt", d),
                        store.pool("yzt", d), prev_zt_n)
        prev_zt_n = m["zt_n"]
        if d in amt.index:
            m["amount"] = round(float(amt.at[d, "amount"]), 0)
            m["amount_ratio"] = float(amt.at[d, "amount_ratio"])
            m["sh_pct"] = float(amt.at[d, "sh_pct"])
        m["date"] = d
        rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 首日没有昨日涨停数, 晋级率缺失, 丢弃
    df = df[df["promote_rate"].notna()].reset_index(drop=True)
    sc = [score(r)[0] for r in df.to_dict("records")]
    df["score"] = sc
    ph = []
    for i, s in enumerate(sc):
        s1 = sc[i - 1] if i >= 1 else None
        s2 = sc[i - 2] if i >= 2 else None
        ph.append(phase(s, s1, s2))
    df["phase"] = ph
    cols = ["date", "score", "phase", "zt_n", "zb_n", "dt_n", "seal_rate", "max_lb",
            "lb_n", "promote_rate", "yzt_avg", "ylb_avg", "big_loss_n",
            "amount", "amount_ratio", "sh_pct", "dragon"]
    return df[[c for c in cols if c in df]]


def latest(hist: pd.DataFrame) -> EmotionDay:
    r = hist.iloc[-1].to_dict()
    s, parts = score(r)
    return EmotionDay(r["date"], r, s, parts, r["phase"])
