"""板块强度: 找出当天最强的板块(主线).

强度 = 涨幅 + 资金(主力净流入额/净流入占比) + 宽度(上涨家数占比)
      + 涨停数 + 强势股占比 + 板块内最高连板 + 3日持续性
各项在候选板块之间做百分位排名后加权, 得分 0~100.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from .data import EMClient
from .emotion import clean

log = logging.getLogger(__name__)

# 非题材类的统计/风格/指数成分板块, 不参与主线判断
NOISE = re.compile(
    "融资融券|深股通|沪股通|创业板综|富时罗素|专精特新|机构重仓|标准普尔|小盘|中盘|大盘|"
    "破增发|破发|QFII|MSCI|深成500|中证500|破净|昨日|微盘|上证|基金重仓|转债标的|HS300|"
    "报预|报首亏|报扭亏|股权分散|超跌|低价股|参股|ST股|市净率|贬值受益|养老金|百元股|次新股|"
    "高成长|风格|微利股|周期股|红利|创业成份|价值股|热股|深证100|行业龙头|股权集中|最近多板|"
    "^B股|AB股|近期|趋势股|IPO受益|权重股|央视50|百日新高|做市|茅指数|证金持股|题材股|券商金股|"
    "宁组合|北交所概念|密集调研|举牌|社保重仓|历史新高|反转股|GDR|央国企改革|西部大开发|"
    "长江三角|深圳特区|AH股|一带一路|并购重组|股权激励|股权转让|中字头|中特估|独角兽|"
    "沪企改革|创投|京津冀|东北振兴|高振幅|高换手|触板|连板|首板|摘帽"
)

WEIGHTS = {
    "pct": 20, "main_net": 15, "main_ratio": 10, "breadth": 10,
    "zt_cnt": 20, "strong_ratio": 10, "max_lb": 10, "ret3": 5,
}


def prepare(boards: pd.DataFrame) -> pd.DataFrame:
    """过滤噪声板块, 去掉申万II/III级完全重复的行业, 计算基础字段."""
    b = boards.copy()
    b = b[~b["name"].astype(str).str.contains(NOISE)]
    b["size"] = b[["up", "down", "flat"]].fillna(0).sum(axis=1)
    b = b[b["size"] >= 8]
    # 行业板块: 同一涨幅+成交额 => II/III 级重复, 保留名字不带 Ⅲ 的
    b["_dup"] = b["pct"].round(2).astype(str) + "_" + b["amount"].round(-4).astype(str)
    b["_lvl3"] = b["name"].astype(str).str.endswith("Ⅲ")
    b = b.sort_values("_lvl3").drop_duplicates("_dup").drop(columns=["_dup", "_lvl3"])
    b["breadth"] = b["up"] / b["size"]
    b["main_net"] = b["main_net"] / 1e8  # 亿
    b["amount"] = b["amount"] / 1e8
    return b.reset_index(drop=True)


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="bottom") * 100


def prelim(b: pd.DataFrame) -> pd.Series:
    """不需要成员数据的初筛分."""
    return (_rank(b["pct"]) * 0.4 + _rank(b["main_net"]) * 0.3
            + _rank(b["main_ratio"]) * 0.15 + _rank(b["breadth"]) * 0.15)


def member_stats(members: pd.DataFrame, zt: pd.DataFrame) -> dict:
    """成员股 -> 涨停数/强势股占比/最高连板/龙头."""
    codes = set(members["code"].astype(str))
    z = zt[zt["c"].isin(codes)] if len(zt) else zt
    n = max(len(members), 1)
    out = {
        "zt_cnt": len(z),
        "strong_ratio": float((members["pct"] >= 5).sum()) / n,
        "max_lb": int(z["lbc"].max()) if len(z) else 0,
        "dragon": "", "dragon_code": "",
    }
    if len(z):
        top = z.sort_values(["lbc", "fbt"], ascending=[False, True]).iloc[0]
        out["dragon"] = f"{top['n']}({int(top['lbc'])}板)"
        out["dragon_code"] = top["c"]
    return out


def rank_sectors(client: EMClient, zt: pd.DataFrame, top_concept: int = 30,
                 top_industry: int = 15, with_ret3: bool = True,
                 member_cache: dict | None = None,
                 live_pct: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """返回 (板块强度表, {板块代码: 成员代码集合}).

    member_cache: 盘中监控传入, 成分股按板块缓存(当天基本不变), 只为新进候选的板块请求.
    live_pct: code->实时涨幅 (涨幅榜), 用于刷新缓存成员的涨幅; 不在榜内视为 <5%.
    """
    zt = clean(zt)
    cb = prepare(client.boards("concept"))
    ib = prepare(client.boards("industry"))
    cb["pre"], ib["pre"] = prelim(cb), prelim(ib)
    # 概念板块里太宽泛的(>400只)不适合做主线, 初筛分打折
    cb.loc[cb["size"] > 400, "pre"] *= 0.7
    cand = pd.concat([cb.nlargest(top_concept, "pre"), ib.nlargest(top_industry, "pre")],
                     ignore_index=True)

    stats, members = [], {}
    for r in cand.itertuples():
        m = member_cache.get(r.bk) if member_cache is not None else None
        if m is None:
            try:
                m = client.board_members(r.bk)
            except RuntimeError as e:
                log.warning("成员获取失败 %s: %s", r.name, e)
                m = pd.DataFrame(columns=["code", "name", "pct"])
            if member_cache is not None and len(m):
                member_cache[r.bk] = m
        members[r.bk] = set(m["code"].astype(str))
        if live_pct is not None and len(m):
            m = m.assign(pct=m["code"].astype(str).map(live_pct).fillna(0.0))
        st = member_stats(m, zt)
        if with_ret3:
            try:
                k = client.kline(f"90.{r.bk}", lmt=4)
                st["ret3"] = (k["close"].iloc[-1] / k["close"].iloc[0] - 1) * 100 if len(k) >= 4 else 0.0
            except RuntimeError:
                st["ret3"] = float("nan")
        else:
            st["ret3"] = float("nan")
        stats.append(st)
    df = pd.concat([cand.reset_index(drop=True), pd.DataFrame(stats)], axis=1)
    # 成分股完全相同的板块(申万II/III级、同名概念)只保留一个
    df["_key"] = df["bk"].map(lambda b: hash(frozenset(members.get(b, ()))) if members.get(b) else b)
    df = df.drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)

    tot = sum(WEIGHTS.values())
    df["score"] = sum(_rank(df[k]) * w for k, w in WEIGHTS.items()) / tot
    # 无涨停的板块不可能是情绪主线
    df.loc[df["zt_cnt"] == 0, "score"] *= 0.6
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["score"] = df["score"].round(1)
    df["theme"] = cluster_themes(df, members)
    return df, members


def cluster_themes(df: pd.DataFrame, members: dict, thr: float = 0.5) -> list[str]:
    """按成分股重叠度把板块聚成"主线". 重叠系数 = |A∩B| / min(|A|,|B|).

    df 需已按 score 降序; 每个簇以得分最高的板块命名.
    """
    heads: list[tuple[str, set]] = []
    out = []
    for r in df.itertuples():
        s = members.get(r.bk, set())
        name = r.name
        for hname, hs in heads:
            if s and hs and len(s & hs) / min(len(s), len(hs)) >= thr:
                name = hname
                break
        else:
            heads.append((r.name, s))
        out.append(name)
    return out


def industry_heat(zt: pd.DataFrame) -> pd.Series:
    """历史可复现的板块热度: 涨停池按行业(hybk)计数. 用于回测."""
    zt = clean(zt)
    return zt.groupby("hybk")["c"].count() if len(zt) else pd.Series(dtype=int)
