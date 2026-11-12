"""回测: 用东财涨停池历史(约20个交易日)检验
1) 个股 core_score 是否能区分后续收益;
2) 情绪分/阶段过滤是否能提高收益.

两种成交假设:
- board (打板): D 日以涨停价买入. 剔除 D 日开盘即涨停(一字/T字)以及
  开盘 1 分钟内封死的秒板 —— 这类票封单巨大, 散户排队基本成交不了,
  不剔除会严重高估收益. 卖出: D+1 开盘 / D+1 收盘.
  仍偏乐观: 实际排队成交的往往是封单弱、次日表现差的那部分.
- next_open (次日接力): D 日收盘后选股, D+1 开盘价买入 (D+1 一字开盘视为买不到),
  因 T+1 只能 D+2 卖出: D+2 开盘 / D+2 收盘. 完全可执行.
统一扣除 0.2% 双边成本+滑点. 只用 D 日收盘已知信息打分, 无未来函数.
"""
from __future__ import annotations

import pandas as pd

from .data import Store, is_final
from .emotion import history as emo_history
from .stocks import CORE_W, core_features, trade_minutes

COST = 0.2  # %


def spearman(a: pd.Series, b: pd.Series) -> float:
    """秩相关(不依赖 scipy)."""
    return a.rank().corr(b.rank())


def limit_pct(code: str) -> float:
    c = str(code)
    if c.startswith(("30", "68")):
        return 20.0
    if c.startswith(("8", "4", "92")):
        return 30.0
    return 10.0


def build_samples(store: Store, emo: pd.DataFrame | None = None) -> pd.DataFrame:
    """每行 = D 日一只涨停股 + 后续 D+1/D+2 的K线收益."""
    emo = emo if emo is not None else emo_history(store)
    days = store.trade_days(45)
    pos = {d: i for i, d in enumerate(days)}
    rows = []
    for d in emo["date"]:
        i = pos.get(d)
        if i is None or i + 1 >= len(days) or not is_final(days[i + 1]):
            continue
        f = core_features(store.pool("zt", d))
        if f.empty:
            continue
        f["date"], f["d1"] = d, days[i + 1]
        f["d2"] = days[i + 2] if i + 2 < len(days) and is_final(days[i + 2]) else ""
        rows.append(f)
    s = pd.concat(rows, ignore_index=True)

    kl = store.klines(list(s[["c", "m"]].drop_duplicates().itertuples(index=False, name=None)))
    nan = float("nan")
    recs = []
    for r in s.itertuples():
        k = kl.get(r.c)
        k = k.set_index("date") if k is not None and len(k) else None
        bar = (lambda d: k.loc[d] if k is not None and d in k.index else None)
        b0, b1, b2 = bar(r.date), bar(r.d1), bar(r.d2)
        x = {"open0": nan, "b_open": nan, "b_close": nan, "b_high": nan,
             "n_entry_ok": False, "n_open": nan, "n_close": nan}
        if b0 is not None:
            x["open0"] = b0["open"]
        if b1 is not None:
            x["b_open"] = (b1["open"] / r.p - 1) * 100
            x["b_close"] = (b1["close"] / r.p - 1) * 100
            x["b_high"] = (b1["high"] / r.p - 1) * 100
            # 次日开盘涨幅接近涨停 => 买不到
            x["n_entry_ok"] = x["b_open"] < limit_pct(r.c) - 0.5
            if b2 is not None and x["n_entry_ok"]:
                x["n_open"] = (b2["open"] / b1["open"] - 1) * 100
                x["n_close"] = (b2["close"] / b1["open"] - 1) * 100
        recs.append(x)
    s = pd.concat([s.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    # 打板可成交性: D 日不是开盘即涨停, 且不是开盘1分钟内封死
    s["board_ok"] = (s["open0"] < s["p"] * 0.998) & (s["fbt"].map(trade_minutes) > 1)
    s = s.merge(emo[["date", "score", "phase"]].rename(columns={"score": "emo"}), on="date")
    return s.reset_index(drop=True)


def mode_frame(s: pd.DataFrame, mode: str) -> pd.DataFrame:
    """统一成 ret_open / ret_close / ret_high 三列."""
    if mode == "board":
        g = s[s["board_ok"]].copy()
        g["ret_open"], g["ret_close"], g["ret_high"] = g["b_open"], g["b_close"], g["b_high"]
    else:
        g = s[s["n_entry_ok"]].copy()
        g["ret_open"], g["ret_close"], g["ret_high"] = g["n_open"], g["n_close"], float("nan")
    return g.dropna(subset=["ret_close"]).reset_index(drop=True)


def _stat(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(g),
        "open%": g["ret_open"].mean() - COST,
        "close%": g["ret_close"].mean() - COST,
        "win_open": (g["ret_open"] - COST > 0).mean(),
        "win_close": (g["ret_close"] - COST > 0).mean(),
    })


def report(s: pd.DataFrame, mode: str = "board", top_k: int = 3,
           exit_col: str = "ret_open") -> dict:
    """exit_col: ret_open(次日/隔日开盘卖, 短线最常用) 或 ret_close."""
    g = mode_frame(s, mode)
    out = {"mode": mode, "samples": len(g), "days": g["date"].nunique()}
    if len(g) < 25:
        return out
    g["q"] = pd.qcut(g["core"].rank(method="first"), 5, labels=["Q1低", "Q2", "Q3", "Q4", "Q5高"])
    out["by_quintile"] = g.groupby("q", observed=True).apply(_stat, include_groups=False)

    ic = {k: spearman(g[f"s_{k}"], g[exit_col]) for k in CORE_W}
    ic["core"] = spearman(g["core"], g[exit_col])
    out["ic"] = pd.Series(ic).round(3)

    daily = g.groupby("date").agg(emo=("emo", "first"), phase=("phase", "first"),
                                  all_ret=(exit_col, "mean"))
    daily[f"top{top_k}"] = (g.sort_values("core", ascending=False).groupby("date")
                            .head(top_k).groupby("date")[exit_col].mean())
    daily = daily.reset_index()
    out["daily"] = daily.round(2)
    out["emo_ic"] = round(spearman(daily["emo"], daily["all_ret"]), 3)
    out["by_phase"] = g.groupby("phase").apply(_stat, include_groups=False)

    def strat(col, mask):
        r = daily.loc[mask, col] - COST
        return {"days": int(mask.sum()), "avg%": r.mean(), "win": (r > 0).mean(),
                "cum%": ((1 + r / 100).prod() - 1) * 100}
    allm = pd.Series(True, index=daily.index)
    ok = ~daily["phase"].isin(["退潮", "冰点"])
    out["strategy"] = pd.DataFrame({
        f"每日top{top_k}": strat(f"top{top_k}", allm),
        f"每日top{top_k}+情绪过滤": strat(f"top{top_k}", ok),
        "全部(基准)": strat("all_ret", allm),
        "全部+情绪过滤": strat("all_ret", ok),
    }).T
    return out
